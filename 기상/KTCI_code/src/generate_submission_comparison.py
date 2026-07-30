from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
REPORT = ROOT / "report" / "KTCI_3Year_Final_Report.md"

SEASONS = ["spring", "summer", "autumn", "winter"]
COMPONENTS = ["thermal", "humidity", "wind", "precipitation", "air_quality"]
COMPONENT_LABELS = {
    "thermal": "온열",
    "humidity": "습도",
    "wind": "바람",
    "precipitation": "강수",
    "air_quality": "대기질",
}
MODEL_LABELS = {
    "KTCI_current": "현재",
    "KTCI_absolute_no_availability": "관측률 제외",
    "KTCI_positive_zero_no_availability": "양의 상관관계 제거 및 관측률 제외",
    "KTCI_2014_adapted": "2014 계절 가중치 adapted benchmark",
}


def save_figure(fig, name, formats, dpi):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        fig.savefig(
            FIGURES / f"{name}.{file_format}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_weight_comparison(config):
    current = pd.read_csv(TABLES / "step4_seasonal_weight_decomposition.csv")
    absolute = pd.read_csv(TABLES / "alternative_absolute_no_availability_weights.csv")
    directional = pd.read_csv(TABLES / "alternative_directional_no_availability_weights.csv")

    merged = current[["season", "component", "data_weight"]].merge(
        absolute[
            [
                "season",
                "component",
                "new_weight_absolute_no_availability",
            ]
        ],
        on=["season", "component"],
        how="inner",
    ).merge(
        directional[
            [
                "season",
                "component",
                "new_weight_directional_no_availability",
            ]
        ],
        on=["season", "component"],
        how="inner",
    )
    merged = merged.rename(
        columns={
            "data_weight": "현재",
            "new_weight_absolute_no_availability": "관측률 제외",
            "new_weight_directional_no_availability": "양의 상관관계 제거 및 관측률 제외",
        }
    )
    merged["season"] = pd.Categorical(merged["season"], categories=SEASONS, ordered=True)
    merged["component"] = pd.Categorical(
        merged["component"],
        categories=COMPONENTS,
        ordered=True,
    )
    merged = merged.sort_values(["season", "component"]).reset_index(drop=True)
    merged["계절"] = merged["season"].map(
        {"spring": "봄", "summer": "여름", "autumn": "가을", "winter": "겨울"}
    )
    merged["구성요소"] = merged["component"].map(COMPONENT_LABELS)
    merged.to_csv(
        TABLES / "submission_three_weight_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    colors = config["style"]["colors"]
    model_colors = [
        colors["data_driven"],
        colors["no_availability"],
        colors["positive_zero_no_availability"],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), sharey=True)
    for ax, season in zip(axes.flat, SEASONS):
        part = merged[merged["season"].eq(season)].set_index("component").reindex(COMPONENTS)
        x = np.arange(len(COMPONENTS))
        width = 0.24
        ax.bar(x - width, part["현재"], width=width, label="현재", color=model_colors[0])
        ax.bar(x, part["관측률 제외"], width=width, label="관측률 제외", color=model_colors[1])
        ax.bar(
            x + width,
            part["양의 상관관계 제거 및 관측률 제외"],
            width=width,
            label="양의 상관관계 제거 및 관측률 제외",
            color=model_colors[2],
        )
        ax.set_xticks(x, [COMPONENT_LABELS[c] for c in COMPONENTS])
        ax.set_title(season)
        ax.set_ylim(0, 0.55)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("세 가지 데이터 기반 KTCI 계절 가중치 비교")
    fig.tight_layout()
    save_figure(
        fig,
        "submission_three_weight_comparison",
        config["style"]["formats"],
        config["style"]["dpi"],
    )
    return merged


def build_performance_comparison(config):
    common = pd.read_csv(TABLES / "three_model_common_sample_performance.csv")
    directional = pd.read_csv(TABLES / "alternative_directional_no_availability_performance.csv")

    current = common[
        common["season"].eq("all") & common["model"].eq("KTCI_current")
    ].iloc[0]
    absolute = common[
        common["season"].eq("all")
        & common["model"].eq("KTCI_absolute_no_availability")
    ].iloc[0]
    positive_zero = common[
        common["season"].eq("all")
        & common["model"].eq("KTCI_positive_zero_no_availability")
    ].iloc[0]
    benchmark = directional[
        directional["season"].eq("all")
        & directional["index"].eq("KTCI_2014_adapted")
    ].iloc[0]

    rows = [
        {
            "model": "KTCI_current",
            "label": MODEL_LABELS["KTCI_current"],
            **current[
                [
                    "n",
                    "spearman",
                    "top_bottom_diff_pp",
                    "test_r2",
                    "test_mae_log",
                ]
            ].to_dict(),
        },
        {
            "model": "KTCI_absolute_no_availability",
            "label": MODEL_LABELS["KTCI_absolute_no_availability"],
            **absolute[
                [
                    "n",
                    "spearman",
                    "top_bottom_diff_pp",
                    "test_r2",
                    "test_mae_log",
                ]
            ].to_dict(),
        },
        {
            "model": "KTCI_positive_zero_no_availability",
            "label": MODEL_LABELS["KTCI_positive_zero_no_availability"],
            **positive_zero[
                [
                    "n",
                    "spearman",
                    "top_bottom_diff_pp",
                    "test_r2",
                    "test_mae_log",
                ]
            ].to_dict(),
        },
        {
            "model": "KTCI_2014_adapted",
            "label": MODEL_LABELS["KTCI_2014_adapted"],
            **benchmark[
                [
                    "n",
                    "spearman",
                    "top_bottom_diff_pp",
                    "test_r2",
                    "test_mae_log",
                ]
            ].to_dict(),
        },
    ]
    performance = pd.DataFrame(rows)
    performance.to_csv(
        TABLES / "submission_four_model_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    colors = config["style"]["colors"]
    palette = [
        colors["data_driven"],
        colors["no_availability"],
        colors["positive_zero_no_availability"],
        colors["benchmark_2014"],
    ]
    short_labels = [
        "현재",
        "관측률 제외",
        "양의 상관관계 제거\n및 관측률 제외",
        "2014 adapted",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("spearman", "Spearman 상관 관계"),
        ("top_bottom_diff_pp", "상·하위 20% 차이(%p)"),
        ("test_r2", "2025 R²"),
        ("test_mae_log", "2025 MAE"),
    ]
    for ax, (column, title) in zip(axes.flat, metrics):
        bars = ax.bar(short_labels, performance[column], color=palette)
        ax.set_title(title)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
        for bar, value in zip(bars, performance[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    fig.suptitle("KTCI 민감도 모델과 2014 benchmark 성능 비교")
    fig.tight_layout()
    save_figure(
        fig,
        "submission_four_model_performance",
        config["style"]["formats"],
        config["style"]["dpi"],
    )
    return performance


def update_report(performance):
    marker = "\n## 민감도 분석 최종 비교\n"
    if REPORT.exists():
        text = REPORT.read_text(encoding="utf-8")
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
    else:
        text = "# 3개년 계절별 KTCI EDA 최종 보고서\n"

    display = performance[
        [
            "label",
            "spearman",
            "top_bottom_diff_pp",
            "test_r2",
            "test_mae_log",
        ]
    ].copy()
    display.columns = [
        "지수",
        "Spearman",
        "상·하위 20% 차이",
        "2025 R²",
        "2025 MAE",
    ]
    section = marker
    section += (
        "현재 가중치 산식과 두 민감도 대안, 2014 계절 가중치 adapted benchmark를 "
        "동일한 관광반응 지표로 비교했습니다.\n\n"
    )
    section += display.to_markdown(index=False, floatfmt=".4f")
    section += (
        "\n\n- 현재: `sqrt(|Spearman| × 10분위 반응 폭 × 관측률)`\n"
        "- 관측률 제외: `sqrt(|Spearman| × 10분위 반응 폭)`\n"
        "- 양의 상관관계 제거 및 관측률 제외: "
        "`sqrt(max(0, -Spearman) × 10분위 반응 폭)`\n"
        "- 2014 adapted: 현재 자료에서 공통으로 계산 가능한 영역에 "
        "2014 계절별 설문 가중치를 재정규화한 benchmark\n"
    )
    REPORT.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    plt.rcParams["font.family"] = config["style"]["font_family"]
    plt.rcParams["axes.unicode_minus"] = False
    build_weight_comparison(config)
    performance = build_performance_comparison(config)
    update_report(performance)
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()
