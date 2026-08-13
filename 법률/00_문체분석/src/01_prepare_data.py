from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, TABLES, ensure_directories


ZIP_PATH = DATA / "raw" / "crime.zip"
TAGGED_MEMBER = "precedents_criminal_tagged.csv"
OUT_PATH = DATA / "interim" / "criminal_panel.parquet"

REASON_START_RE = re.compile(
    r"(?:【\s*이\s*유\s*】|\[\s*이\s*유\s*\]|〔\s*이\s*유\s*〕|"
    r"〈\s*이\s*유\s*〉|(?:^|\n)\s*이\s*유\s*(?=\n))",
    re.IGNORECASE | re.MULTILINE,
)
REASON_END_RE = re.compile(
    r"\n\s*(?:【|\[|〔|〈)?\s*(?:참\s*조\s*조\s*문|참\s*조\s*판\s*례|"
    r"판\s*사|대\s*법\s*관|관\s*여\s*법\s*관|재\s*판\s*장)\s*(?:】|\]|〕|〉)?",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\f\v]+")
MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = html.unescape(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    text = MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def extract_reason(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    match = REASON_START_RE.search(text)
    if not match:
        return "", False
    start = match.end()
    tail = text[start:]
    end_match = REASON_END_RE.search(tail)
    if end_match:
        tail = tail[: end_match.start()]
    tail = normalize_text(tail)
    return tail, len(tail) >= 100


def corrected_year(date_iso: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw_year = pd.to_numeric(date_iso.astype("string").str.slice(0, 4), errors="coerce")
    dangi = raw_year.between(4200, 4399)
    year = raw_year.where(~dangi, raw_year - 2333)
    return year.astype("Int64"), dangi.fillna(False)


def main() -> None:
    ensure_directories()
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Input archive not found: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH) as archive:
        if TAGGED_MEMBER not in archive.namelist():
            raise KeyError(f"{TAGGED_MEMBER} is missing from {ZIP_PATH.name}")
        with archive.open(TAGGED_MEMBER) as stream:
            df = pd.read_csv(stream, dtype="string", low_memory=False)

    required = {
        "precedent_id",
        "decision_date_iso",
        "court_name",
        "body_format",
        "full_text",
        "rule_court_instance",
        "rule_crime_group",
        "rule_document_focus",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    df["year"], df["date_corrected_from_dangi"] = corrected_year(df["decision_date_iso"])
    df["full_text_clean"] = df["full_text"].map(normalize_text)
    extracted = df["full_text_clean"].map(extract_reason)
    df["reason_text"] = extracted.map(lambda x: x[0])
    df["reason_extracted"] = extracted.map(lambda x: x[1])
    # The corrected heading rule extracts reasons for more than 99% of usable
    # judgments in every decade. Keep one consistent reasons-only scope.
    df["analysis_text_source"] = "reason"
    df["analysis_text"] = df["reason_text"]

    df["full_text_chars"] = df["full_text_clean"].str.len().fillna(0).astype("int64")
    df["analysis_chars"] = df["analysis_text"].str.len().fillna(0).astype("int64")
    valid_year = df["year"].between(1950, 2026)
    df["has_usable_text"] = df["analysis_chars"] >= 100
    df["main_period"] = df["year"].between(1980, 2025)

    eligible_lengths = df.loc[valid_year & df["has_usable_text"], "analysis_chars"]
    q999 = float(eligible_lengths.quantile(0.999))
    df["extreme_length_flag"] = df["analysis_chars"] > max(100_000, q999)
    df["prepare_included"] = valid_year & df["has_usable_text"]

    df.to_parquet(OUT_PATH, index=False, compression="zstd")

    audit = pd.DataFrame(
        [
            ("rows", len(df)),
            ("unique_precedent_id", df["precedent_id"].nunique()),
            ("duplicate_precedent_id_rows", int(df["precedent_id"].duplicated().sum())),
            ("usable_text_rows", int(df["has_usable_text"].sum())),
            ("missing_or_short_text_rows", int((~df["has_usable_text"]).sum())),
            ("reason_extracted_rows", int(df["reason_extracted"].sum())),
            ("dangi_date_corrections", int(df["date_corrected_from_dangi"].sum())),
            ("extreme_length_flags", int(df["extreme_length_flag"].sum())),
            ("min_corrected_year", int(df.loc[valid_year, "year"].min())),
            ("max_corrected_year", int(df.loc[valid_year, "year"].max())),
            ("rows_2026", int((df["year"] == 2026).sum())),
            ("analysis_length_q999", q999),
        ],
        columns=["item", "value"],
    )
    audit.to_csv(TABLES / "data_quality.csv", index=False, encoding="utf-8-sig")

    year_table = (
        df.loc[valid_year]
        .groupby("year", observed=True)
        .agg(
            n=("precedent_id", "size"),
            usable_text_n=("has_usable_text", "sum"),
            reason_extracted_n=("reason_extracted", "sum"),
            third_instance_share=("rule_court_instance", lambda s: (s == "third").mean()),
            median_full_text_chars=("full_text_chars", "median"),
            median_analysis_chars=("analysis_chars", "median"),
        )
        .reset_index()
    )
    year_table["reason_extracted_share"] = (
        year_table["reason_extracted_n"] / year_table["n"]
    )
    year_table.to_csv(TABLES / "year_counts.csv", index=False, encoding="utf-8-sig")

    comp = df.loc[valid_year].copy()
    comp["decade"] = (comp["year"].astype(int) // 10) * 10
    decade = (
        comp.groupby("decade", observed=True)
        .agg(
            n=("precedent_id", "size"),
            third_instance_share=("rule_court_instance", lambda s: (s == "third").mean()),
            reason_extracted_share=("reason_extracted", "mean"),
            median_analysis_chars=("analysis_chars", "median"),
            court_count=("court_name", "nunique"),
        )
        .reset_index()
    )
    decade.to_csv(TABLES / "decade_composition.csv", index=False, encoding="utf-8-sig")

    qc_cols = [
        "precedent_id",
        "case_name",
        "case_number",
        "year",
        "court_name",
        "analysis_text_source",
        "full_text_chars",
        "analysis_chars",
        "extreme_length_flag",
    ]
    qc = df.loc[df["extreme_length_flag"] | ~df["has_usable_text"], qc_cols]
    qc.to_csv(TABLES / "text_qc_flags.csv", index=False, encoding="utf-8-sig")

    print(f"Prepared {len(df):,} rows -> {OUT_PATH}")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
