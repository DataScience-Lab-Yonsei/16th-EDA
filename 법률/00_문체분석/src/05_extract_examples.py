from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import DATA, EXAMPLES, ensure_directories


def compact_excerpt(text: str, limit: int = 650) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sentence_end = max(cut.rfind("."), cut.rfind("다."), cut.rfind("요."))
    if sentence_end >= int(limit * 0.6):
        cut = cut[: sentence_end + 1]
    return cut.rstrip() + "…"


def main() -> None:
    ensure_directories()
    panel = pd.read_parquet(DATA / "interim" / "criminal_panel.parquet", columns=["precedent_id", "analysis_text"])
    metrics = pd.read_parquet(DATA / "processed" / "document_metrics.parquet")
    df = metrics.merge(panel, on="precedent_id", how="left", validate="one_to_one")
    subset = df.loc[
        df["main_analysis"]
        & df["year"].isin([1985, 2005, 2025])
        & df["rule_court_instance"].eq("third")
        & df["rule_crime_group"].eq("재산범죄")
    ].copy()
    selected = []
    for year, group in subset.groupby("year"):
        for metric in ["noun_share", "sent_eojeol_mean"]:
            sd = group[metric].std(ddof=1)
            group[f"z_{metric}"] = (group[metric] - group[metric].median()) / (sd if sd else 1)
        group["distance"] = np.sqrt(group["z_noun_share"] ** 2 + group["z_sent_eojeol_mean"] ** 2)
        selected.append(group.nsmallest(1, "distance").iloc[0])

    lines = [
        "# 동일 조건 판결문 예시",
        "",
        "3심·재산범죄 안에서 각 연도의 명사 비율과 문장 길이가 중앙값에 가까운 판결문을 골랐다. 예시는 수치 해석을 돕기 위한 것이며 전체 판결문을 대표하지 않는다.",
        "",
    ]
    for row in selected:
        lines.extend(
            [
                f"## {int(row['year'])}년 · {row['case_name']}",
                "",
                f"- 판례 ID: `{row['precedent_id']}`",
                f"- 법원: {row['court_name']}",
                f"- 명사 비율: {row['noun_share'] * 100:.1f}%",
                f"- 동사 비율: {row['verb_share'] * 100:.1f}%",
                f"- 조사 비율: {row['particle_share'] * 100:.1f}%",
                f"- 문장당 어절 수: {row['sent_eojeol_mean']:.1f}",
                "",
                "> " + compact_excerpt(row["analysis_text"]),
                "",
            ]
        )
    (EXAMPLES / "matched_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(selected)} matched examples")


if __name__ == "__main__":
    main()
