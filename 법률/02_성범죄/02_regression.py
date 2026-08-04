#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NEW_ANAL_SEX, OUTPUT


def run_logit(data: pd.DataFrame, formula: str, label: str) -> dict:
    d = data.copy()
    try:
        model = smf.logit(formula, data=d).fit(disp=0, maxiter=200, method="lbfgs")
    except Exception as e:  # noqa: BLE001
        return {"label": label, "error": str(e)}
    ci = model.conf_int()
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "OR": np.exp(model.params.values),
            "CI_low": np.exp(ci[0].values),
            "CI_high": np.exp(ci[1].values),
            "p": model.pvalues.values,
        }
    )
    table["sig_05"] = np.where(table["p"] < 0.05, "유의", "비유의")
    table.to_csv(OUTPUT / f"logit_{label}.csv", index=False, encoding="utf-8-sig")
    print(f"\n===== {label} n={int(model.nobs)} R²={model.prsquared:.3f} =====")
    print(table.to_string(index=False))
    return {
        "label": label,
        "n": int(model.nobs),
        "prsquared": float(model.prsquared),
        "llr_pvalue": float(model.llr_pvalue),
        "table": table,
    }


def main() -> None:
    if not NEW_ANAL_SEX.exists():
        raise FileNotFoundError(
            f"{NEW_ANAL_SEX} 없음. 먼저 실행: python3 code/01_preprocess_tag.py"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"[02] load {NEW_ANAL_SEX}")
    df = pd.read_csv(NEW_ANAL_SEX)

    # outcome TRUE/FALSE → 1/0
    df["outcome"] = df["outcome"].map(
        {True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0, 1: 1, 0: 0}
    )
    df = df.dropna(subset=["lawyer", "outcome"]).copy()
    df = df[df["lawyer"].isin(["public", "lawyer", "lawfirm"])].copy()
    df["lawyer"] = pd.Categorical(
        df["lawyer"], categories=["public", "lawyer", "lawfirm"]
    )
    df["sex_type"] = pd.Categorical(
        df["sex_type"], categories=["indecent", "rape", "camera"]
    )
    df["agree"] = df["agree"].astype(int)

    # 기술통계
    df.groupby("lawyer", observed=False)["outcome"].agg(["count", "mean"]).to_csv(
        OUTPUT / "02_desc_lawyer_x_outcome.csv", encoding="utf-8-sig"
    )
    df.groupby("sex_type", observed=False)["outcome"].agg(["count", "mean"]).to_csv(
        OUTPUT / "02_desc_sextype_x_outcome.csv", encoding="utf-8-sig"
    )
    df.groupby("agree")["outcome"].agg(["count", "mean"]).to_csv(
        OUTPUT / "02_desc_agree_x_outcome.csv", encoding="utf-8-sig"
    )
    (
        df.groupby(["sex_type", "lawyer"], observed=False)["outcome"]
        .agg(["count", "mean"])
        .reset_index()
        .to_csv(OUTPUT / "02_desc_layer_sextype_lawyer.csv", index=False, encoding="utf-8-sig")
    )

    results = []

    # (8) 주 모형
    results.append(
        run_logit(
            df,
            "outcome ~ C(lawyer, Treatment(reference='public')) + C(sex_type) "
            "+ agree + first_offender + same_record",
            "A_minimal",
        )
    )
    results.append(
        run_logit(
            df,
            "outcome ~ C(lawyer, Treatment(reference='public')) + C(sex_type) "
            "+ agree + reflection + first_offender + same_record + recovery + minor_victim",
            "B_extended",
        )
    )
    df_y = df.dropna(subset=["year"]).copy()
    results.append(
        run_logit(
            df_y,
            "outcome ~ C(lawyer, Treatment(reference='public')) + C(sex_type) "
            "+ agree + first_offender + same_record + year",
            "C_with_year",
        )
    )
    results.append(
        run_logit(
            df,
            "outcome ~ C(lawyer, Treatment(reference='lawyer')) + C(sex_type) "
            "+ agree + first_offender + same_record",
            "E_ref_private",
        )
    )

    # (9) 합의 ~ 변호사
    results.append(
        run_logit(
            df,
            "agree ~ C(lawyer, Treatment(reference='public')) + C(sex_type) "
            "+ first_offender + same_record + reflection",
            "D_agree_on_lawyer",
        )
    )

    # (10) 집유만
    df["suspended"] = (df["punishment"] == "suspended").astype(int)
    results.append(
        run_logit(
            df,
            "suspended ~ C(lawyer, Treatment(reference='public')) + C(sex_type) "
            "+ agree + first_offender + same_record",
            "F_suspended_only",
        )
    )

    # (10) 죄유형별 변호사 outcome 표는 이미 layer CSV

    fit_rows = []
    for r in results:
        if "error" in r:
            fit_rows.append({"model": r["label"], "error": r["error"]})
        else:
            fit_rows.append(
                {
                    "model": r["label"],
                    "n": r["n"],
                    "pseudo_r2": round(r["prsquared"], 4),
                    "llr_p": r["llr_pvalue"],
                }
            )
    pd.DataFrame(fit_rows).to_csv(
        OUTPUT / "02_model_fit_summary.csv", index=False, encoding="utf-8-sig"
    )

    # 짧은 보고서
    lawyer_tbl = df.groupby("lawyer", observed=False)["outcome"].agg(["count", "mean"])
    fit_df = pd.DataFrame(fit_rows)
    lines = [
        "# 성범죄 회귀 결과 요약",
        "",
        f"입력: `{NEW_ANAL_SEX.name}` · n(변호사 비결측)={len(df)}",
        "",
        "## 변호사별 outcome 비율",
        "",
        "```",
        lawyer_tbl.to_string(),
        "```",
        "",
        "## 모형 적합",
        "",
        "```",
        fit_df.to_string(index=False),
        "```",
        "",
        "## 해석 주의",
        "- 소표본 · 양형구간 단순화 · 키워드 태깅 · 선택편향",
        "- 변호사 효과가 비유의여도 ‘효과 없음’ 단정은 금지 (검정력 부족 가능)",
        "",
        "상세 OR 표: `output/logit_*.csv`",
        "",
    ]
    (OUTPUT / "02_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "n_reg": len(df),
        "outcome_rate": float(df["outcome"].mean()),
        "lawyer_counts": df["lawyer"].value_counts().to_dict(),
        "models": fit_rows,
    }
    (OUTPUT / "02_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[02] done → {OUTPUT}")


if __name__ == "__main__":
    main()
