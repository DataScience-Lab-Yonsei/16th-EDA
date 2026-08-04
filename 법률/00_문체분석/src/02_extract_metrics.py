from __future__ import annotations

import importlib.metadata
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from tqdm import tqdm

from common import DATA, TABLES, ensure_directories


INPUT = DATA / "interim" / "criminal_panel.parquet"
OUTPUT = DATA / "processed" / "document_metrics.parquet"
PARTS = DATA / "interim" / "metric_parts"
VARIANTS = Path(__file__).resolve().parents[1] / "config" / "variant_pairs.csv"
BATCH_SIZE = 200

LING_PREFIXES = {"N", "V", "M", "I", "J", "E", "X", "C"}
PREDICATE_TAGS = {"VV", "VA", "VX", "VCP", "VCN"}
CONTENT_PREFIXES = {"N", "V", "M", "I"}
HANJA_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
NONSPACE_RE = re.compile(r"\s+")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def metric_record(sentences: list[list], text: str, variants: list[dict]) -> dict:
    tokens = [token for sent in sentences for token in sent]
    ling = [t for t in tokens if t.tag and t.tag[0] in LING_PREFIXES]
    denom = len(ling)
    nouns = sum(t.tag.startswith("N") for t in ling)
    strict_verbs = sum(t.tag == "VV" for t in ling)
    predicates = sum(t.tag in PREDICATE_TAGS for t in ling)
    particles = sum(t.tag.startswith("J") for t in ling)
    endings = sum(t.tag.startswith("E") for t in ling)
    nominalizers = sum(t.tag == "ETN" for t in ling)
    adnominals = sum(t.tag == "ETM" for t in ling)
    content = sum(t.tag[0] in CONTENT_PREFIXES for t in ling)

    valid_sentences = [
        sent for sent in sentences if any(t.tag and not t.tag.startswith("S") for t in sent)
    ]
    sentence_count = len(valid_sentences)
    eojeol_count = sum(
        len({t.word_position for t in sent if t.tag and t.tag[0] in LING_PREFIXES})
        for sent in valid_sentences
    )
    nonspace_chars = len(NONSPACE_RE.sub("", text))
    hanja_count = len(HANJA_RE.findall(text))

    out = {
        "mp_tokens": denom,
        "all_tokens": len(tokens),
        "sentence_count": sentence_count,
        "eojeol_count": eojeol_count,
        "noun_count": nouns,
        "verb_count": strict_verbs,
        "predicate_count": predicates,
        "particle_count": particles,
        "ending_count": endings,
        "nominalizer_count": nominalizers,
        "adnominal_count": adnominals,
        "content_count": content,
        "noun_share": safe_div(nouns, denom),
        "verb_share": safe_div(strict_verbs, denom),
        "predicate_share": safe_div(predicates, denom),
        "particle_share": safe_div(particles, denom),
        "ending_share": safe_div(endings, denom),
        "nominality": safe_div(nouns, nouns + predicates),
        "nominalizer_per1k": safe_div(nominalizers * 1000, denom),
        "adnominal_per1k": safe_div(adnominals * 1000, denom),
        "sent_eojeol_mean": safe_div(eojeol_count, sentence_count),
        "sent_char_mean": safe_div(nonspace_chars, sentence_count),
        "hanja_count": hanja_count,
        "hanja_share": safe_div(hanja_count, nonspace_chars),
    }
    for pair in variants:
        out[f"variant_{pair['key']}_old"] = len(pair["old_re"].findall(text))
        out[f"variant_{pair['key']}_new"] = len(pair["new_re"].findall(text))
    return out


def main() -> None:
    ensure_directories()
    PARTS.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(f"Run 01_prepare_data.py first: {INPUT}")

    panel = pd.read_parquet(INPUT)
    panel = panel.loc[panel["prepare_included"]].reset_index(drop=True)
    variant_df = pd.read_csv(VARIANTS, encoding="utf-8")
    variants = []
    for row in variant_df.to_dict("records"):
        row["old_re"] = re.compile(str(row["old_pattern"]))
        row["new_re"] = re.compile(str(row["new_pattern"]))
        variants.append(row)

    workers = min(6, max(2, (os.cpu_count() or 4) - 1))
    kiwi = Kiwi(num_workers=workers)
    started = time.time()
    parts: list[Path] = []

    for batch_no, start in enumerate(tqdm(range(0, len(panel), BATCH_SIZE), desc="Kiwi batches")):
        part_path = PARTS / f"metrics_{batch_no:04d}.parquet"
        parts.append(part_path)
        if part_path.exists():
            continue
        batch = panel.iloc[start : start + BATCH_SIZE]
        texts = batch["analysis_text"].fillna("").astype(str).tolist()
        records = []
        try:
            analyses = kiwi.tokenize(texts, split_sents=True)
            for precedent_id, text, sentences in zip(batch["precedent_id"], texts, analyses):
                rec = {"precedent_id": precedent_id, "metric_error": ""}
                rec.update(metric_record(sentences, text, variants))
                records.append(rec)
        except Exception as exc:
            # Preserve progress and isolate malformed documents if a bulk batch fails.
            for precedent_id, text in zip(batch["precedent_id"], texts):
                try:
                    rec = {"precedent_id": precedent_id, "metric_error": ""}
                    rec.update(metric_record(kiwi.tokenize(text, split_sents=True), text, variants))
                except Exception as item_exc:
                    rec = {
                        "precedent_id": precedent_id,
                        "metric_error": f"{type(item_exc).__name__}: {item_exc}",
                    }
                records.append(rec)
        pd.DataFrame(records).to_parquet(part_path, index=False, compression="zstd")

    metrics = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    keep_meta = [
        "precedent_id",
        "case_name",
        "case_number",
        "year",
        "court_name",
        "body_format",
        "rule_court_instance",
        "rule_crime_name",
        "rule_crime_group",
        "rule_document_focus",
        "analysis_text_source",
        "analysis_chars",
        "extreme_length_flag",
        "reason_extracted",
        "main_period",
    ]
    out = panel[keep_meta].merge(metrics, on="precedent_id", how="left", validate="one_to_one")
    out["metric_included"] = (
        out["metric_error"].fillna("").eq("")
        & out["mp_tokens"].fillna(0).ge(150)
        & out["year"].between(1950, 2026)
    )
    out["main_analysis"] = out["metric_included"] & out["year"].between(1980, 2025)
    out.to_parquet(OUTPUT, index=False, compression="zstd")

    audit = pd.DataFrame(
        [
            ("kiwipiepy_version", importlib.metadata.version("kiwipiepy")),
            ("kiwi_workers", workers),
            ("prepared_rows", len(panel)),
            ("metric_error_rows", int(out["metric_error"].fillna("").ne("").sum())),
            ("mp_tokens_ge_150", int(out["metric_included"].sum())),
            ("main_1980_2025", int(out["main_analysis"].sum())),
            ("elapsed_seconds", round(time.time() - started, 2)),
        ],
        columns=["item", "value"],
    )
    audit.to_csv(TABLES / "metric_audit.csv", index=False, encoding="utf-8-sig")
    errors = out.loc[out["metric_error"].fillna("").ne(""), ["precedent_id", "metric_error"]]
    errors.to_csv(TABLES / "metric_errors.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(out):,} document metrics -> {OUTPUT}")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
