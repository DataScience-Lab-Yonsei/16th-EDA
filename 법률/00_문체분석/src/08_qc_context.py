from __future__ import annotations

import re

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi

from common import DATA, EXAMPLES, TABLES, ensure_directories


METRICS_INPUT = DATA / "processed" / "document_metrics.parquet"
PANEL_INPUT = DATA / "interim" / "criminal_panel.parquet"
VARIANTS_INPUT = DATA.parent / "config" / "variant_pairs.csv"
SAMPLE_PER_YEAR = 20
RNG_SEED = 20260723
LING_PREFIXES = {"N", "V", "M", "I", "J", "E", "X", "C"}


def recalc_counts(kiwi: Kiwi, text: str) -> dict:
    sentences = kiwi.tokenize(text, split_sents=True)
    tokens = [token for sent in sentences for token in sent]
    ling = [token for token in tokens if token.tag and token.tag[0] in LING_PREFIXES]
    valid_sentences = [
        sent for sent in sentences if any(t.tag and not t.tag.startswith("S") for t in sent)
    ]
    return {
        "recalc_mp_tokens": len(ling),
        "recalc_sentence_count": len(valid_sentences),
        "recalc_noun_count": sum(t.tag.startswith("N") for t in ling),
        "recalc_verb_count": sum(t.tag == "VV" for t in ling),
        "recalc_particle_count": sum(t.tag.startswith("J") for t in ling),
        "recalc_symbol_share": 1 - len(ling) / max(len(tokens), 1),
    }


def audit_1980s(metrics: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = metrics.loc[
        metrics["main_analysis"] & metrics["year"].between(1980, 1989)
    ].copy()
    sampled = pd.concat(
        [
            group.sample(
                n=min(SAMPLE_PER_YEAR, len(group)),
                random_state=RNG_SEED + int(year),
            )
            for year, group in frame.groupby("year", observed=True)
        ],
        ignore_index=True,
    )
    sampled = sampled.merge(
        panel[["precedent_id", "analysis_text"]],
        on="precedent_id",
        how="left",
        validate="one_to_one",
    )

    kiwi = Kiwi(num_workers=4)
    recalculated = pd.DataFrame(
        [recalc_counts(kiwi, str(text)) for text in sampled["analysis_text"]]
    )
    audit = pd.concat([sampled.reset_index(drop=True), recalculated], axis=1)
    audit["counts_match"] = (
        audit["mp_tokens"].eq(audit["recalc_mp_tokens"])
        & audit["sentence_count"].eq(audit["recalc_sentence_count"])
        & audit["noun_count"].eq(audit["recalc_noun_count"])
        & audit["verb_count"].eq(audit["recalc_verb_count"])
        & audit["particle_count"].eq(audit["recalc_particle_count"])
    )
    audit["very_long_sentence"] = audit["sent_eojeol_mean"].gt(100)
    audit["few_boundaries_for_long_text"] = audit["sentence_count"].le(2) & audit[
        "mp_tokens"
    ].gt(500)
    audit["high_hanja"] = audit["hanja_share"].gt(0.01)
    audit["boundary_risk"] = (
        audit["very_long_sentence"] | audit["few_boundaries_for_long_text"]
    )
    audit["snippet"] = (
        audit["analysis_text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 300)
    )

    summary = pd.DataFrame(
        [
            ("sample_size", len(audit)),
            ("years_covered", audit["year"].nunique()),
            ("count_recalculation_mismatches", (~audit["counts_match"]).sum()),
            ("boundary_risk_documents", audit["boundary_risk"].sum()),
            ("very_long_sentence_documents", audit["very_long_sentence"].sum()),
            ("few_boundaries_for_long_text_documents", audit["few_boundaries_for_long_text"].sum()),
            ("high_hanja_documents", audit["high_hanja"].sum()),
            ("median_symbol_share", audit["recalc_symbol_share"].median()),
        ],
        columns=["item", "value"],
    )
    return audit, summary


def occurrence_context(text: str, start: int, end: int, width: int = 90) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    context = re.sub(r"\s+", " ", text[left:right]).strip()
    return ("…" if left > 0 else "") + context + ("…" if right < len(text) else "")


def inside_quotation(text: str, position: int) -> bool:
    prefix = text[:position]
    if prefix.count('"') % 2 == 1:
        return True
    for opening, closing in [("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』")]:
        if prefix.rfind(opening) > prefix.rfind(closing):
            return True
    return False


def context_candidates(
    panel: pd.DataFrame, pattern: re.Pattern[str], pair_key: str, form: str
) -> pd.DataFrame:
    rows = []
    for row in panel.itertuples(index=False):
        text = str(row.analysis_text)
        match = pattern.search(text)
        if not match:
            continue
        context = occurrence_context(text, match.start(), match.end())
        nested = False
        if pair_key == "negative_lexeme":
            nested = bool(re.search(r"하지\s*(?:아니하|않)", context))
        rows.append(
            {
                "key": pair_key,
                "form": form,
                "year": int(row.year),
                "precedent_id": row.precedent_id,
                "case_name": row.case_name,
                "matched_text": match.group(0),
                "match_start": match.start(),
                "nested_in_long_negative_construction": nested,
                "context": context,
            }
        )
    return pd.DataFrame(rows)


def spread_sample(candidates: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    if len(candidates) <= n:
        return candidates.sort_values(["year", "precedent_id"])
    ordered = candidates.sort_values(["year", "precedent_id"]).reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return ordered.iloc[np.unique(indices)].copy()


def audit_variants(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(VARIANTS_INPUT, encoding="utf-8")
    text_lookup = panel.set_index("precedent_id")["analysis_text"].astype(str).to_dict()
    samples = []
    summaries = []
    for pair in pairs.to_dict("records"):
        old_candidates = context_candidates(
            panel, re.compile(str(pair["old_pattern"])), str(pair["key"]), "old"
        )
        new_candidates = context_candidates(
            panel, re.compile(str(pair["new_pattern"])), str(pair["key"]), "new"
        )
        old_sample = spread_sample(old_candidates)
        new_sample = spread_sample(new_candidates)
        for sample in [old_sample, new_sample]:
            sample["inside_quotation"] = [
                inside_quotation(text_lookup[row.precedent_id], int(row.match_start))
                for row in sample.itertuples(index=False)
            ]
        samples.extend([old_sample, new_sample])
        for form, candidates, sample in [
            ("old", old_candidates, old_sample),
            ("new", new_candidates, new_sample),
        ]:
            summaries.append(
                {
                    "key": pair["key"],
                    "label_old": pair["label_old"],
                    "label_new": pair["label_new"],
                    "form": form,
                    "documents_with_match": len(candidates),
                    "first_year": candidates["year"].min() if not candidates.empty else np.nan,
                    "last_year": candidates["year"].max() if not candidates.empty else np.nan,
                    "nested_long_negative_share": (
                        candidates["nested_in_long_negative_construction"].mean()
                        if not candidates.empty
                        else np.nan
                    ),
                    "sample_inside_quotation_share": (
                        sample["inside_quotation"].mean()
                        if not sample.empty
                        else np.nan
                    ),
                }
            )
    return pd.concat(samples, ignore_index=True), pd.DataFrame(summaries)


def write_flagged_markdown(audit: pd.DataFrame) -> None:
    flagged = audit.loc[audit["boundary_risk"] | audit["high_hanja"]].copy()
    flagged = flagged.sort_values(
        ["boundary_risk", "sent_eojeol_mean", "hanja_share"],
        ascending=False,
    ).head(20)
    lines = [
        "# 1980년대 형태소·문장 경계 위험 표본",
        "",
        "1980–1989년에서 연도별 20건씩 뽑은 200건 가운데, 문장 경계가 매우 길거나 한자 비율이 높은 문서를 우선 검토하도록 정리했다.",
        "",
    ]
    for row in flagged.itertuples(index=False):
        lines.extend(
            [
                f"## {int(row.year)}년 · {row.case_name}",
                "",
                (
                    f"- 판례 ID: `{row.precedent_id}` · 문장당 어절: "
                    f"{row.sent_eojeol_mean:.1f} · 한자 비율: {row.hanja_share * 100:.2f}%"
                ),
                f"- 문장 경계 위험: {'예' if row.boundary_risk else '아니오'}",
                "",
                f"> {row.snippet}…",
                "",
            ]
        )
    (EXAMPLES / "qc_1980s_flagged_examples.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ensure_directories()
    metrics = pd.read_parquet(METRICS_INPUT)
    panel = pd.read_parquet(PANEL_INPUT)
    panel_main = panel.loc[
        panel["precedent_id"].isin(metrics.loc[metrics["main_analysis"], "precedent_id"])
    ].copy()

    audit, audit_summary = audit_1980s(metrics, panel)
    contexts, context_summary = audit_variants(panel_main)
    audit.to_csv(TABLES / "qc_1980s_sample.csv", index=False, encoding="utf-8-sig")
    audit_summary.to_csv(
        TABLES / "qc_1980s_summary.csv", index=False, encoding="utf-8-sig"
    )
    contexts.to_csv(
        TABLES / "variant_context_audit.csv", index=False, encoding="utf-8-sig"
    )
    context_summary.to_csv(
        TABLES / "variant_context_summary.csv", index=False, encoding="utf-8-sig"
    )
    write_flagged_markdown(audit)

    print("1980s QC summary:")
    print(audit_summary.to_string(index=False))
    print("\nVariant context summary:")
    print(context_summary.to_string(index=False))


if __name__ == "__main__":
    main()
