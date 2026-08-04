#!/usr/bin/env python
# coding: utf-8
"""교통범죄(음주운전·도주치상) 사건의 전처리와 탐색적 회귀분석.

한서현 님의 사기범죄 분석 흐름을 재현하되, 다음 두 한계를 명시적으로
분리한다.

1. 단순 음주운전 양형기준은 2023-07-01 시행분부터 적용되므로 과거 사건에
   현재 기준을 소급하지 않는다.
2. 판결문 한 건에 여러 죄명·피고인·심급이 섞일 수 있으므로 자동 추출값과
   함께 원문 근거와 검수 플래그를 출력한다.

주요 산출물은 CSV, PNG, Markdown이다. 원자료는 변경하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from statsmodels.stats.outliers_influence import variance_inflation_factor


DEFAULT_INPUT = Path("EDA_data_Preprocessed.txt")
DEFAULT_OUTPUT = Path(".")

OFFICIAL_SOURCES = {
    "current_traffic_guideline": (
        "https://sc.scourt.go.kr/sc/krsc/criterion/"
        "criterion_35/traffic_change_01.jsp"
    ),
    "traffic_guideline_2012": (
        "https://sc.scourt.go.kr/sc/krsc/criterion/"
        "past/traffic_2012/traffic_01.jsp"
    ),
    "traffic_guideline_2016": (
        "https://sc.scourt.go.kr/sc/krsc/criterion/"
        "past/traffic_2016/traffic_01.jsp"
    ),
    "traffic_guideline_2020_pdf": (
        "https://sc.scourt.go.kr/sc/krsc/pdf/"
        "F19.Crimes_of_Traffic%282020%29.pdf"
    ),
    "traffic_guideline_2023_pdf": (
        "https://sc.scourt.go.kr/sc/krsc/pdf/"
        "F19.Crimes_of_Traffic%282023%29.pdf"
    ),
    "guideline_explanation": (
        "https://sc.scourt.go.kr/sc/krsc/pdf/sc_explan_doc.pdf"
    ),
}


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text)


def compact_text(value: object) -> str:
    return re.sub(r"\s+", " ", normalize_text(value)).strip()


def truthy(value: object) -> bool:
    return str(value).strip().upper() in {"T", "TRUE", "1"}


def docket_year(case_number: object) -> float:
    match = re.search(r"(19|20)\d{2}", compact_text(case_number))
    return float(match.group(0)) if match else np.nan


def get_section(text: str, start_patterns: Iterable[str], max_chars: int = 6000) -> str:
    """첫 시작 표제부터 다음 대괄호 표제 전까지 반환한다."""
    normalized = normalize_text(text)
    starts: list[tuple[int, int]] = []
    for pattern in start_patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if match:
            starts.append((match.start(), match.end()))
    if not starts:
        return ""
    _, end = min(starts)
    tail = normalized[end : end + max_chars]
    next_header = re.search(r"\n?\s*【[^】]{1,30}】", tail)
    if next_header:
        tail = tail[: next_header.start()]
    return compact_text(tail)


def extract_order(text: str) -> str:
    return get_section(text, [r"【\s*주\s*문\s*】"], max_chars=5000)


def extract_sentencing_section(text: str) -> str:
    return get_section(
        text,
        [
            r"【\s*양형의\s*이유\s*】",
            r"【\s*양형\s*이유\s*】",
            r"(?:^|\n)\s*\d*\.?\s*양형의\s*이유",
            r"(?:^|\n)\s*양형\s*이유",
            r"(?:^|\n)\s*양형\s*사유",
        ],
        max_chars=10000,
    )


def context(text: str, start: int, end: int, window: int = 170) -> str:
    return compact_text(text[max(0, start - window) : min(len(text), end + window)])


MONTH_PATTERNS = [
    re.compile(r"징역\s*(\d+)\s*년\s*(?:(\d+)\s*(?:개월|월))?"),
    re.compile(r"징역\s*(\d+)\s*(?:개월|월)"),
    re.compile(r"금고\s*(\d+)\s*년\s*(?:(\d+)\s*(?:개월|월))?"),
    re.compile(r"금고\s*(\d+)\s*(?:개월|월)"),
]
FINE_PATTERN = re.compile(
    r"벌금\s*(?:금\s*)?([0-9,]+)\s*(억|천만|백만|만)?\s*원"
)


def amount_to_won(number: str, unit: str | None) -> float:
    value = float(number.replace(",", ""))
    multiplier = {
        "억": 100_000_000,
        "천만": 10_000_000,
        "백만": 1_000_000,
        "만": 10_000,
        None: 1,
    }[unit]
    return value * multiplier


def sentence_candidates(text: str, order: str) -> list[dict[str, object]]:
    full = normalize_text(text)
    candidates: list[dict[str, object]] = []
    regions = [("order", order), ("full_text", full)]
    for region_name, region in regions:
        if not region:
            continue
        for pattern_no, pattern in enumerate(MONTH_PATTERNS):
            for match in pattern.finditer(region):
                groups = match.groups()
                if pattern_no in {0, 2}:
                    months = int(groups[0]) * 12 + int(groups[1] or 0)
                else:
                    months = int(groups[0])
                snippet = context(region, match.start(), match.end())
                score = 100 if region_name == "order" else 0
                score += 30 if re.search(r"선고|처한다|정한다", snippet) else 0
                score += 20 if re.search(r"원심|제1심|항소심", snippet) else 0
                score += 25 if "집행유예" in snippet else 0
                score -= 80 if re.search(r"법정형|처단형|형량범위", snippet) else 0
                score -= 50 if re.search(
                    r"징역\s*\d+\s*(?:년|개월|월)\s*(?:이상|이하)",
                    snippet,
                ) else 0
                candidates.append(
                    {
                        "kind": "custodial",
                        "months": float(months),
                        "fine_won": np.nan,
                        "region": region_name,
                        "score": score,
                        "suspended_nearby": "집행유예" in snippet,
                        "snippet": snippet,
                    }
                )
        for match in FINE_PATTERN.finditer(region):
            snippet = context(region, match.start(), match.end())
            score = 100 if region_name == "order" else 0
            score += 30 if re.search(r"선고|처한다|정한다", snippet) else 0
            score += 20 if re.search(r"원심|제1심|항소심", snippet) else 0
            score -= 80 if re.search(r"법정형|처단형|형량범위", snippet) else 0
            candidates.append(
                {
                    "kind": "fine",
                    "months": np.nan,
                    "fine_won": amount_to_won(match.group(1), match.group(2)),
                    "region": region_name,
                    "score": score,
                    "suspended_nearby": False,
                    "snippet": snippet,
                }
            )
    # order와 full_text 양쪽에서 잡힌 동일 후보는 원문 근거가 더 좋은 order를 우선한다.
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for candidate in sorted(candidates, key=lambda x: float(x["score"]), reverse=True):
        key = (
            candidate["kind"],
            candidate["months"],
            candidate["fine_won"],
            candidate["snippet"],
        )
        unique.setdefault(key, candidate)
    return list(unique.values())


def select_sentence(
    candidates: list[dict[str, object]], punishment: object
) -> dict[str, object]:
    punishment_text = compact_text(punishment)
    ranked: list[dict[str, object]] = []
    for candidate in candidates:
        item = candidate.copy()
        if punishment_text in {"imprisonment", "suspended", "both"}:
            item["score"] = float(item["score"]) + (
                35 if item["kind"] == "custodial" else -20
            )
        elif punishment_text == "fine":
            item["score"] = float(item["score"]) + (
                35 if item["kind"] == "fine" else -20
            )
        if punishment_text == "suspended" and item["suspended_nearby"]:
            item["score"] = float(item["score"]) + 25
        ranked.append(item)
    if not ranked:
        return {
            "sentence_kind": "",
            "sentence_months": np.nan,
            "fine_won": np.nan,
            "sentence_region": "",
            "sentence_score": np.nan,
            "sentence_suspended_nearby": False,
            "sentence_excerpt": "",
            "sentence_candidates": "",
        }
    ranked.sort(key=lambda x: float(x["score"]), reverse=True)
    best = ranked[0]
    return {
        "sentence_kind": best["kind"],
        "sentence_months": best["months"],
        "fine_won": best["fine_won"],
        "sentence_region": best["region"],
        "sentence_score": best["score"],
        "sentence_suspended_nearby": best["suspended_nearby"],
        "sentence_excerpt": best["snippet"],
        "sentence_candidates": json.dumps(
            [
                {
                    "kind": x["kind"],
                    "months": x["months"],
                    "fine_won": x["fine_won"],
                    "region": x["region"],
                    "score": x["score"],
                    "snippet": x["snippet"][:240],
                }
                for x in ranked[:5]
            ],
            ensure_ascii=False,
        ),
    }


def extract_bac(text: str) -> tuple[float, str, str]:
    normalized = normalize_text(text)
    matches: list[tuple[float, int, int]] = []
    patterns = [
        r"혈중\s*알코올\s*농도(?:가|는|를)?\s*(?:약\s*)?([0-9]+(?:\.[0-9]+)?)\s*%",
        r"혈중알코올농도(?:가|는|를)?\s*(?:약\s*)?([0-9]+(?:\.[0-9]+)?)\s*퍼센트",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            value = float(match.group(1))
            if 0.01 <= value <= 1:
                matches.append((value, match.start(), match.end()))
    if not matches:
        return np.nan, "", ""
    matches.sort(key=lambda x: x[1])
    value, start, end = matches[0]
    all_values = "|".join(f"{x[0]:.3f}" for x in matches)
    return value, all_values, context(normalized, start, end)


def bac_type(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if 0.03 <= value < 0.08:
        return "BAC_0.03_to_0.08"
    if 0.08 <= value < 0.2:
        return "BAC_0.08_to_0.20"
    if value >= 0.2:
        return "BAC_0.20_plus"
    return "below_0.03_or_invalid"


def extract_injury_weeks(text: str) -> tuple[float, str]:
    normalized = normalize_text(text)
    patterns = [
        r"(?:약\s*)?(\d+)\s*주(?:간)?의?\s*(?:치료|가료)",
        r"(?:치료|가료)\s*(?:기간\s*)?(?:약\s*)?(\d+)\s*주",
    ]
    hits: list[tuple[int, int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            hits.append((int(match.group(1)), match.start(), match.end()))
    if not hits:
        return np.nan, ""
    # 사건 내 피해가 여러 명이면 가장 중한 치료기간을 사용한다.
    value, start, end = max(hits, key=lambda x: x[0])
    return float(value), context(normalized, start, end)


def keyword_tag(
    text: str,
    positive_patterns: Iterable[str],
    negative_patterns: Iterable[str] = (),
    window: int = 90,
) -> tuple[float, str]:
    """긍정 표현 주변에 부정 표현이 없을 때 1, 명시적 부정만 있으면 0."""
    normalized = normalize_text(text)
    positive_hits: list[str] = []
    for pattern in positive_patterns:
        for match in re.finditer(pattern, normalized, flags=re.I):
            snippet = context(normalized, match.start(), match.end(), window)
            if not any(re.search(neg, snippet, flags=re.I) for neg in negative_patterns):
                positive_hits.append(snippet)
    if positive_hits:
        return 1.0, " || ".join(dict.fromkeys(positive_hits[:3]))
    negative_hits: list[str] = []
    for pattern in negative_patterns:
        for match in re.finditer(pattern, normalized, flags=re.I):
            negative_hits.append(context(normalized, match.start(), match.end(), window))
    if negative_hits:
        return 0.0, " || ".join(dict.fromkeys(negative_hits[:3]))
    return 0.0, ""


FACTOR_PATTERNS = {
    "settlement": (
        [
            r"피해자(?:들)?(?:과|와)\s*(?:원만히\s*)?합의",
            r"합의에\s*이르",
            r"처벌을\s*원하지\s*(?:아니|않)",
            r"처벌불원",
        ],
        [
            r"합의하지\s*못",
            r"합의되지\s*(?:아니|않)",
            r"합의에\s*이르지\s*못",
            r"처벌을\s*원하",
        ],
    ),
    "damage_recovery": (
        [
            r"피해(?:가|를)?\s*(?:상당\s*부분\s*)?회복",
            r"피해\s*회복을\s*위",
            r"공탁",
            r"자동차종합보험",
            r"종합보험에\s*가입",
            r"보험금.{0,20}지급",
        ],
        [
            r"피해(?:가|를)?\s*회복하지\s*못",
            r"피해\s*회복이\s*이루어지지\s*(?:아니|않)",
            r"보험에\s*가입하지\s*(?:아니|않)",
        ],
    ),
    "remorse": (
        [r"진지하게\s*반성", r"잘못을\s*(?:깊이\s*)?반성", r"반성하고\s*있"],
        [r"반성하지\s*(?:아니|않)", r"진지한\s*반성\s*없"],
    ),
    "no_prior": (
        [
            r"형사처벌\s*전력이\s*없",
            r"처벌받은\s*전력이\s*없",
            r"아무런\s*전과가\s*없",
            r"초범",
        ],
        [],
    ),
    "same_prior": (
        [
            r"동종\s*전과",
            r"동종\s*범행",
            r"음주운전.{0,20}(?:전과|전력|처벌)",
            r"누범",
        ],
        [r"동종\s*전과가\s*없", r"음주운전.{0,20}전력이\s*없"],
    ),
    "concealment": (
        [
            r"증거(?:를\s*)?(?:은폐|인멸)",
            r"범행(?:을\s*)?(?:은폐|은닉)",
            r"기록(?:을\s*)?삭제",
            r"측정.{0,20}(?:방해|회피)",
        ],
        [],
    ),
    "high_road_risk": (
        [
            r"교통상의\s*위험이\s*매우\s*높",
            r"사고를\s*(?:야기|발생)",
            r"교통사고를\s*(?:야기|발생)",
            r"무면허운전",
            r"난폭운전",
        ],
        [r"교통상의\s*위험이\s*매우\s*낮"],
    ),
}


def classify_guideline(row: pd.Series) -> dict[str, object]:
    subgroup = row["crime_subgroup"]
    year = row["docket_year"]
    decision_date = pd.to_datetime(row["decision_date_iso"], errors="coerce")
    abandonment = bool(row["abandonment"])
    result: dict[str, object] = {
        "guideline_version": "",
        "guideline_type": "",
        "guideline_applicable": False,
        "guideline_applicability_note": "",
        "basic_low_months": np.nan,
        "basic_high_months": np.nan,
        "basic_low_fine_won": np.nan,
        "basic_high_fine_won": np.nan,
    }
    if subgroup == "hit_run_injury":
        if pd.isna(year):
            result["guideline_applicability_note"] = "공소연도 추정 불가"
            return result
        if year < 2016:
            version, low, high = "2012", 8, 18
        elif year < 2020:
            version, low, high = "2016", 8, 18
        elif year < 2023:
            version, low, high = "2020", 8, 30
        else:
            version, low, high = "2023", 10, 30
        if abandonment:
            low, high = 24, 48
        result.update(
            {
                "guideline_version": version,
                "guideline_type": (
                    "hit_run_injury_abandonment"
                    if abandonment
                    else "hit_run_injury"
                ),
                "guideline_applicable": True,
                "guideline_applicability_note": (
                    "사건번호의 연도를 공소연도 대용치로 사용; 시행월 경계사건은 검수 필요"
                ),
                "basic_low_months": float(low),
                "basic_high_months": float(high),
            }
        )
        return result

    # 단순 음주운전 기준은 2023-07-01 시행. 사건번호의 연도만으로 2023년
    # 사건의 공소일이 시행 전후인지 확정할 수 없으므로 2024년 이후만 확정한다.
    if pd.isna(year) or year < 2023:
        result["guideline_applicability_note"] = (
            "단순 음주운전 양형기준 시행(2023-07-01) 전 사건"
        )
        return result
    if year == 2023:
        result.update(
            {
                "guideline_version": "2023_uncertain",
                "guideline_applicability_note": (
                    "2023년 사건: 정확한 공소제기일 확인 전에는 적용 여부 불확실"
                ),
            }
        )
        return result
    value = row["bac"]
    if pd.isna(value):
        result.update(
            {
                "guideline_version": "2023",
                "guideline_applicability_note": "혈중알코올농도 미추출",
            }
        )
        return result
    if 0.03 <= value < 0.08:
        kind, low, high, fine_low, fine_high = (
            "drunk_BAC_0.03_to_0.08",
            0,
            8,
            2_000_000,
            4_000_000,
        )
    elif 0.08 <= value < 0.2:
        kind, low, high, fine_low, fine_high = (
            "drunk_BAC_0.08_to_0.20",
            8,
            16,
            5_000_000,
            8_000_000,
        )
    elif value >= 0.2:
        kind, low, high, fine_low, fine_high = (
            "drunk_BAC_0.20_plus",
            18,
            36,
            10_000_000,
            17_000_000,
        )
    else:
        result.update(
            {
                "guideline_version": "2023",
                "guideline_applicability_note": "0.03% 미만 또는 비정상 농도",
            }
        )
        return result
    result.update(
        {
            "guideline_version": "2023",
            "guideline_type": kind,
            "guideline_applicable": True,
            "guideline_applicability_note": (
                "사건번호 연도가 2024년 이후이고 BAC 자동추출에 성공"
            ),
            "basic_low_months": float(low),
            "basic_high_months": float(high),
            "basic_low_fine_won": float(fine_low),
            "basic_high_fine_won": float(fine_high),
        }
    )
    return result


def add_reduced_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["below_basic"] = np.nan
    custodial = (
        data["guideline_applicable"]
        & data["sentence_kind"].eq("custodial")
        & data["sentence_months"].notna()
        & data["basic_low_months"].notna()
    )
    data.loc[custodial, "below_basic"] = (
        data.loc[custodial, "sentence_months"]
        < data.loc[custodial, "basic_low_months"]
    ).astype(int)
    fine = (
        data["guideline_applicable"]
        & data["sentence_kind"].eq("fine")
        & data["fine_won"].notna()
        & data["basic_low_fine_won"].notna()
    )
    data.loc[fine, "below_basic"] = (
        data.loc[fine, "fine_won"] < data.loc[fine, "basic_low_fine_won"]
    ).astype(int)

    known_punishment = data["punishment"].isin(
        ["suspended", "imprisonment", "both", "fine"]
    )
    data["outcome_suspended"] = np.where(
        known_punishment,
        data["punishment"].eq("suspended").astype(int),
        np.nan,
    )
    # 한서현 방식: 집행유예이거나, 징역형이면서 기본영역 하한보다 낮으면 1.
    comparable = data["punishment"].eq("suspended") | (
        data["punishment"].eq("imprisonment") & data["below_basic"].notna()
    )
    data["outcome_han"] = np.where(
        comparable,
        (
            data["punishment"].eq("suspended")
            | (
                data["punishment"].eq("imprisonment")
                & data["below_basic"].eq(1)
            )
        ).astype(int),
        np.nan,
    )
    data["outcome_lenient_expanded"] = np.where(
        known_punishment,
        (
            data["punishment"].eq("suspended") | data["below_basic"].eq(1)
        ).astype(int),
        np.nan,
    )
    return data


def build_analysis_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = raw.copy()
    case_source = data["case_name"].fillna("").astype(str)
    tag_source = data["tag_crime_name_manual"].fillna("").astype(str)
    source = case_source + " " + tag_source
    data["is_drunk_driving"] = source.str.contains("음주운전", regex=False)
    data["is_hit_run_injury"] = (
        source.str.contains(r"도주치상|도주차량", regex=True)
        & ~source.str.contains("도주치사", regex=False)
    )
    candidate = data[
        data["tag_crime_group_manual"].eq("traffic")
        & data["guilty"].eq("Guilty")
        & (data["is_drunk_driving"] | data["is_hit_run_injury"])
    ].copy()
    candidate["sentencing_reason_flag"] = candidate[
        "tag_sentencing_reason_present"
    ].map(truthy)
    candidate["quality_rank"] = (
        candidate["tag_sentence_source"].eq("current_order").astype(int) * 10
        + candidate["tag_document_focus"].eq("facts_sentencing").astype(int) * 5
        + candidate["sentencing_reason_flag"].astype(int) * 3
        + candidate["lawyer"].notna().astype(int)
        + candidate["punishment"].notna().astype(int)
    )
    candidate = candidate.sort_values(
        ["precedent_id", "quality_rank"], ascending=[True, False]
    )
    precedent_duplicates = candidate[
        candidate.duplicated("precedent_id", keep=False)
    ].copy()
    precedent_duplicates["duplicate_reason"] = "same_precedent_id"
    candidate = candidate.drop_duplicates("precedent_id", keep="first").copy()
    identity_columns = ["case_number", "decision_date_iso", "court_name"]
    identity_duplicates = candidate[
        candidate.duplicated(identity_columns, keep=False)
    ].copy()
    identity_duplicates["duplicate_reason"] = "same_case_date_court"
    candidate = candidate.sort_values(
        identity_columns + ["quality_rank"],
        ascending=[True, True, True, False],
    ).drop_duplicates(identity_columns, keep="first")
    duplicates = pd.concat(
        [precedent_duplicates, identity_duplicates], ignore_index=True
    )
    # 주된 죄명은 수동 태그보다 정식 사건명을 우선한다. 일부 태그에는
    # 재물손괴 후 미조치 사건도 '도주차량'으로 넓게 적혀 있기 때문이다.
    primary_hit_run = (
        candidate["case_name"].fillna("").str.contains(
            r"도주치상|도주차량", regex=True
        )
        & ~candidate["case_name"].fillna("").str.contains(
            "도주치사", regex=False
        )
    )
    candidate["crime_subgroup"] = np.where(
        primary_hit_run, "hit_run_injury", "drunk_driving"
    )
    candidate["drunk_overlap"] = (
        primary_hit_run & candidate["is_drunk_driving"]
    )
    candidate["docket_year"] = candidate["case_number"].map(docket_year)
    candidate["decision_year"] = pd.to_datetime(
        candidate["decision_date_iso"], errors="coerce"
    ).dt.year

    records: list[dict[str, object]] = []
    for _, row in candidate.iterrows():
        text = normalize_text(row["full_text"])
        order = extract_order(text)
        sentencing_section = extract_sentencing_section(text)
        reason_section = get_section(
            text, [r"【\s*이\s*유\s*】"], max_chars=14000
        )
        selected = select_sentence(
            sentence_candidates(text, order), row.get("punishment")
        )
        bac, all_bac, bac_excerpt = extract_bac(text)
        injury, injury_excerpt = extract_injury_weeks(text)
        abandonment, abandonment_excerpt = keyword_tag(
            text,
            [r"피해자(?:를|들을)?.{0,80}유기하고\s*도주", r"유기\s*도주"],
            [],
            window=140,
        )
        if sentencing_section:
            factor_text = sentencing_section
            factor_scope = "sentencing_section"
            factor_evidence_reliable = True
        elif truthy(row.get("tag_sentencing_reason_present")) and reason_section:
            factor_text = reason_section
            factor_scope = "tagged_sentencing_reason"
            factor_evidence_reliable = True
        else:
            factor_text = text
            factor_scope = "full_text_screening_only"
            factor_evidence_reliable = False
        factors: dict[str, object] = {}
        for factor, (positive, negative) in FACTOR_PATTERNS.items():
            value, snippet = keyword_tag(
                factor_text, positive, negative, window=120
            )
            factors[f"{factor}_raw"] = value
            factors[factor] = value if factor_evidence_reliable else np.nan
            factors[f"{factor}_excerpt"] = snippet
        case_name = compact_text(row["case_name"])
        serious_other = bool(
            re.search(
                r"도주치사|위험운전치사(?!상)|살인|강제추행|성폭력|"
                r"특수상해|마약|사기|공무집행방해|운전자폭행",
                case_name,
            )
        )
        multi_charge = bool(re.search(r"[·ㆍ,]", case_name))
        multi_defendant = bool(
            re.search(r"피고인\s*[12]|피고인들", order or text[:3500])
        )
        record = row.to_dict()
        record.update(
            {
                "order_excerpt": order[:1200],
                "sentencing_section": sentencing_section[:5000],
                "factor_scope": factor_scope,
                "factor_evidence_reliable": int(factor_evidence_reliable),
                "bac": bac,
                "bac_all": all_bac,
                "bac_type": bac_type(bac),
                "bac_excerpt": bac_excerpt,
                "injury_weeks_max": injury,
                "injury_excerpt": injury_excerpt,
                "abandonment": int(abandonment),
                "abandonment_excerpt": abandonment_excerpt,
                "multi_charge": int(multi_charge),
                "multi_defendant": int(multi_defendant),
                "serious_other_charge": int(serious_other),
                **selected,
                **factors,
            }
        )
        records.append(record)
    analysis = pd.DataFrame(records)
    guidelines = analysis.apply(classify_guideline, axis=1, result_type="expand")
    analysis = pd.concat([analysis.reset_index(drop=True), guidelines], axis=1)
    analysis = add_reduced_outcomes(analysis)

    analysis["sentence_match"] = (
        (
            analysis["punishment"].isin(["imprisonment", "suspended", "both"])
            & analysis["sentence_kind"].eq("custodial")
            & analysis["sentence_months"].notna()
        )
        | (
            analysis["punishment"].eq("fine")
            & analysis["sentence_kind"].eq("fine")
            & analysis["fine_won"].notna()
        )
    ).astype(int)
    analysis["analysis_eligible_broad"] = (
        analysis["lawyer"].isin(["public", "lawyer", "lawfirm"])
        & analysis["punishment"].isin(
            ["suspended", "imprisonment", "both", "fine"]
        )
    ).astype(int)
    analysis["analysis_eligible"] = (
        analysis["analysis_eligible_broad"].eq(1)
        & analysis["serious_other_charge"].eq(0)
        & analysis["multi_defendant"].eq(0)
        & (
            analysis["sentencing_reason_flag"]
            | analysis["tag_document_focus"].isin(
                ["facts_sentencing", "mixed"]
            )
        )
    ).astype(int)
    review_reasons: list[str] = []
    for _, row in analysis.iterrows():
        reasons: list[str] = []
        if not row["sentence_match"]:
            reasons.append("선고형 자동추출 불일치")
        if row["multi_charge"]:
            reasons.append("병합·복수 죄명")
        if row["multi_defendant"]:
            reasons.append("복수 피고인")
        if row["serious_other_charge"]:
            reasons.append("중대 비대상 죄명 병합")
        if not row["factor_evidence_reliable"]:
            reasons.append("신뢰 가능한 양형이유 구간 없음")
        if row["crime_subgroup"] == "drunk_driving" and pd.isna(row["bac"]):
            reasons.append("BAC 미추출")
        if row["crime_subgroup"] == "hit_run_injury" and pd.isna(
            row["injury_weeks_max"]
        ):
            reasons.append("상해기간 미추출")
        review_reasons.append("; ".join(reasons))
    analysis["review_reasons"] = review_reasons
    analysis["manual_review_required"] = analysis["review_reasons"].ne("").astype(int)
    return analysis, duplicates


def model_result_table(model, model_name: str, outcome: str, n: int) -> pd.DataFrame:
    params = model.params
    ci = model.conf_int()
    exp_params = np.exp(np.clip(params.values, -700, 700))
    exp_low = np.exp(np.clip(ci.iloc[:, 0].values, -700, 700))
    exp_high = np.exp(np.clip(ci.iloc[:, 1].values, -700, 700))
    result = pd.DataFrame(
        {
            "model": model_name,
            "outcome": outcome,
            "n": n,
            "term": params.index,
            "coefficient": params.values,
            "std_error": model.bse.values,
            "OR": exp_params,
            "CI_lower": exp_low,
            "CI_upper": exp_high,
            "p_value": model.pvalues.values,
            "pseudo_r2": getattr(model, "prsquared", np.nan),
            "aic": model.aic,
        }
    )
    result["significant_05"] = result["p_value"] < 0.05
    return result


def safe_logit(
    data: pd.DataFrame,
    formula: str,
    model_name: str,
    outcome: str,
) -> tuple[pd.DataFrame, dict[str, object], object | None]:
    columns = sorted(
        set(
            re.findall(
                r"\b(?:outcome_[a-z_]+|lawyer|crime_subgroup|decision_year|"
                r"settlement|damage_recovery|same_prior|remorse|no_prior)\b",
                formula,
            )
        )
    )
    model_data = data[columns].dropna().copy()
    diagnostics: dict[str, object] = {
        "model": model_name,
        "outcome": outcome,
        "formula": formula,
        "n": len(model_data),
        "events": (
            int(model_data[outcome].sum()) if outcome in model_data else np.nan
        ),
        "non_events": (
            int(len(model_data) - model_data[outcome].sum())
            if outcome in model_data
            else np.nan
        ),
        "status": "",
        "warning": "",
    }
    if len(model_data) < 12 or model_data[outcome].nunique() < 2:
        diagnostics["status"] = "not_fitted"
        diagnostics["warning"] = "표본 또는 사건 수 부족"
        return pd.DataFrame(), diagnostics, None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.logit(formula=formula, data=model_data).fit(
                disp=False, maxiter=300
            )
            warning_text = " | ".join(str(x.message) for x in caught)
        converged = bool(model.mle_retvals.get("converged"))
        diagnostics["status"] = "fitted" if converged else "nonconverged"
        diagnostics["warning"] = warning_text
        diagnostics["converged"] = converged
        return (
            model_result_table(model, model_name, outcome, len(model_data)),
            diagnostics,
            model,
        )
    except Exception as exc:  # small-sample separation is an expected failure mode
        diagnostics["status"] = "failed"
        diagnostics["warning"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), diagnostics, None


def compute_vif(model, model_name: str) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame()
    design = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    rows = []
    for index, name in enumerate(design.columns):
        try:
            value = variance_inflation_factor(design.values, index)
        except Exception:
            value = np.nan
        rows.append(
            {
                "model": model_name,
                "term": name,
                "VIF": value,
                "tolerance": 1 / value if value and np.isfinite(value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pairwise_fisher(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
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
                "outcome": outcome,
                "comparison": f"{group} vs public",
                "n": len(subset),
                "event_group": int(matrix[0][0]),
                "non_event_group": int(matrix[0][1]),
                "event_public": int(matrix[1][0]),
                "non_event_public": int(matrix[1][1]),
                "OR": odds_ratio,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def run_models(
    analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = analysis[analysis["analysis_eligible"].eq(1)].copy()
    broad = analysis[analysis["analysis_eligible_broad"].eq(1)].copy()
    for frame in [eligible, broad]:
        for column in ["lawyer", "crime_subgroup"]:
            frame[column] = frame[column].astype("category")
    specifications = [
        (
            eligible,
            "M0_primary_suspension_lawyer",
            "outcome_suspended ~ C(lawyer, Treatment(reference='public'))",
            "outcome_suspended",
        ),
        (
            eligible,
            "M1_primary_suspension_adjusted",
            "outcome_suspended ~ C(lawyer, Treatment(reference='public'))"
            " + C(crime_subgroup) + decision_year",
            "outcome_suspended",
        ),
        (
            eligible,
            "M2_primary_suspension_factors",
            "outcome_suspended ~ C(lawyer, Treatment(reference='public'))"
            " + C(crime_subgroup) + settlement + same_prior",
            "outcome_suspended",
        ),
        (
            broad,
            "S1_broad_suspension_lawyer",
            "outcome_suspended ~ C(lawyer, Treatment(reference='public'))",
            "outcome_suspended",
        ),
        (
            broad,
            "S2_broad_suspension_adjusted",
            "outcome_suspended ~ C(lawyer, Treatment(reference='public'))"
            " + C(crime_subgroup) + decision_year",
            "outcome_suspended",
        ),
        (
            eligible,
            "M3_han_reproduction",
            "outcome_han ~ C(lawyer, Treatment(reference='public'))"
            " + C(crime_subgroup)",
            "outcome_han",
        ),
        (
            eligible,
            "M4_settlement",
            "settlement ~ C(lawyer, Treatment(reference='public'))"
            " + C(crime_subgroup) + same_prior",
            "settlement",
        ),
    ]
    result_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    vif_frames: list[pd.DataFrame] = []
    for model_data, name, formula, outcome in specifications:
        result, diagnostic, model = safe_logit(
            model_data, formula, name, outcome
        )
        diagnostics.append(diagnostic)
        if not result.empty:
            result_frames.append(result)
        vif = compute_vif(model, name)
        if not vif.empty:
            vif_frames.append(vif)
    results = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame()
    )
    diagnostics_frame = pd.DataFrame(diagnostics)
    vifs = (
        pd.concat(vif_frames, ignore_index=True)
        if vif_frames
        else pd.DataFrame()
    )
    fishers = pd.concat(
        [
            pairwise_fisher(eligible, "outcome_suspended"),
            pairwise_fisher(eligible, "settlement"),
        ],
        ignore_index=True,
    )
    return results, diagnostics_frame, vifs, fishers


def make_descriptives(analysis: pd.DataFrame) -> pd.DataFrame:
    eligible = analysis[analysis["analysis_eligible"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    for group_columns in [
        ["crime_subgroup"],
        ["lawyer"],
        ["crime_subgroup", "lawyer"],
    ]:
        for keys, group in eligible.groupby(group_columns, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record: dict[str, object] = {
                "grouping": " × ".join(group_columns),
                "n": len(group),
                "suspended_n": int(group["outcome_suspended"].sum()),
                "suspended_rate": group["outcome_suspended"].mean(),
                "settlement_n": int(group["settlement"].sum()),
                "settlement_rate": group["settlement"].mean(),
                "guideline_comparable_n": int(group["below_basic"].notna().sum()),
                "below_basic_n": int(group["below_basic"].fillna(0).sum()),
            }
            for column, key in zip(group_columns, keys):
                record[column] = key
            rows.append(record)
    return pd.DataFrame(rows)


def wilson_interval(events: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = events / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - half, center + half


def create_figures(
    raw: pd.DataFrame,
    analysis: pd.DataFrame,
    model_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

    counts = pd.Series(
        {
            "전체": len(raw),
            "교통범죄": int(raw["tag_crime_group_manual"].eq("traffic").sum()),
            "교통 유죄": int(
                (
                    raw["tag_crime_group_manual"].eq("traffic")
                    & raw["guilty"].eq("Guilty")
                ).sum()
            ),
            "배정 죄명\n고유사건": len(analysis),
            "회귀 가능": int(analysis["analysis_eligible"].sum()),
            "기준 비교 가능": int(analysis["below_basic"].notna().sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#A9B6C6", "#6E8FAF", "#3E6B91", "#2A9D8F", "#E9C46A", "#E76F51"]
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_title("교통범죄 분석 표본 흐름", fontweight="bold")
    ax.set_ylabel("사건 수")
    ax.set_ylim(0, max(counts.values) * 1.12)
    fig.tight_layout()
    fig.savefig(output_dir / "fig01_sample_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    eligible = analysis[analysis["analysis_eligible"].eq(1)]
    rate_rows = []
    for lawyer, group in eligible.groupby("lawyer", observed=True):
        events = int(group["outcome_suspended"].sum())
        n = len(group)
        low, high = wilson_interval(events, n)
        rate_rows.append(
            {
                "lawyer": lawyer,
                "rate": events / n,
                "low": low,
                "high": high,
                "n": n,
            }
        )
    rates = pd.DataFrame(rate_rows)
    order = ["public", "lawyer", "lawfirm"]
    rates["lawyer"] = pd.Categorical(rates["lawyer"], order, ordered=True)
    rates = rates.sort_values("lawyer")
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    labels = {
        "public": "국선",
        "lawyer": "개인변호사",
        "lawfirm": "법무법인",
    }
    x = np.arange(len(rates))
    ax.errorbar(
        x,
        rates["rate"],
        yerr=[
            rates["rate"] - rates["low"],
            rates["high"] - rates["rate"],
        ],
        fmt="o",
        markersize=10,
        capsize=6,
        color="#2A6F97",
        ecolor="#89A7BE",
    )
    ax.set_xticks(x, [labels.get(str(x), str(x)) for x in rates["lawyer"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("집행유예 비율 (95% Wilson CI)")
    ax.set_title("변호사 유형별 집행유예 비율", fontweight="bold")
    for idx, row in rates.reset_index(drop=True).iterrows():
        ax.text(idx, min(0.96, row["high"] + 0.06), f"n={int(row['n'])}", ha="center")
    fig.tight_layout()
    fig.savefig(
        output_dir / "fig02_suspension_by_lawyer.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    if not model_results.empty:
        forest = model_results[
            model_results["term"].str.contains(
                r"C\(lawyer.*T\.(?:lawyer|lawfirm)\]", regex=True
            )
        ].copy()
        forest = forest[
            forest["model"].isin(
                [
                    "M0_primary_suspension_lawyer",
                    "M1_primary_suspension_adjusted",
                ]
            )
        ]
        if not forest.empty:
            forest["label"] = (
                forest["model"].replace(
                    {
                        "M0_primary_suspension_lawyer": "무보정",
                        "M1_primary_suspension_adjusted": "죄명·연도 보정",
                    }
                )
                + " · "
                + np.where(
                    forest["term"].str.contains("lawfirm"),
                    "법무법인 vs 국선",
                    "개인 vs 국선",
                )
            )
            forest = forest.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["OR", "CI_lower", "CI_upper"]
            )
            if not forest.empty:
                fig, ax = plt.subplots(figsize=(9, 5.8))
                y = np.arange(len(forest))
                ax.errorbar(
                    forest["OR"],
                    y,
                    xerr=[
                        forest["OR"] - forest["CI_lower"],
                        forest["CI_upper"] - forest["OR"],
                    ],
                    fmt="o",
                    capsize=5,
                    color="#264653",
                    ecolor="#7B9E9D",
                )
                ax.axvline(1, color="#C44536", linestyle="--", linewidth=1)
                ax.set_xscale("log")
                ax.set_xticks([0.1, 1, 10, 100])
                ax.set_xticklabels(["0.1", "1", "10", "100"])
                ax.set_yticks(y, forest["label"])
                ax.set_xlabel("오즈비 (로그 눈금, 95% CI)")
                ax.set_title("집행유예 로지스틱 회귀: 변호사 효과", fontweight="bold")
                ax.invert_yaxis()
                fig.tight_layout()
                fig.savefig(
                    output_dir / "fig03_lawyer_forest.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.close(fig)


def codebook_frame() -> pd.DataFrame:
    rows = [
        ("crime_subgroup", "주된 배정 죄명", "hit_run_injury / drunk_driving"),
        ("lawyer", "변호사 유형", "public / lawyer / lawfirm"),
        ("punishment", "기존 전처리 형종", "suspended / imprisonment / both / fine"),
        ("sentence_months", "자동 추출 자유형 개월 수", "주문 우선, 원심·제1심 표현 보조"),
        ("fine_won", "자동 추출 벌금액", "원 단위"),
        ("bac", "첫 혈중알코올농도", "% 숫자, 복수값은 bac_all과 원문 검수"),
        ("injury_weeks_max", "최대 치료기간", "주 단위"),
        ("settlement", "합의·처벌불원 표현", "0/1 자동 태깅"),
        ("damage_recovery", "피해회복·공탁·종합보험 표현", "0/1 자동 태깅"),
        ("remorse", "진지한 반성 표현", "0/1 자동 태깅"),
        ("no_prior", "형사처벌 전력 없음·초범", "0/1 자동 태깅"),
        ("same_prior", "동종전과·음주운전 전력·누범", "0/1 자동 태깅"),
        ("concealment", "증거은폐·인멸·측정회피", "0/1 자동 태깅"),
        ("high_road_risk", "높은 교통위험 대용표현", "0/1 자동 태깅"),
        ("below_basic", "기본영역 하한 미만", "적용 가능한 사건만 0/1, 나머지 NA"),
        ("outcome_han", "한서현 방식 감경 결과", "집행유예 또는 징역형 기준하회"),
        ("outcome_suspended", "전 기간 비교용 결과", "집행유예=1, 다른 확인 형종=0"),
        ("analysis_eligible_broad", "광의 민감도 표본", "변호사 유형과 형종 확인"),
        (
            "analysis_eligible",
            "주 분석 표본",
            "광의 표본 중 중대 병합죄·복수피고인·순수 법리사건 제외",
        ),
        ("manual_review_required", "수동 검수 필요", "자동추출·병합죄·양형이유 부재 등"),
    ]
    return pd.DataFrame(rows, columns=["variable", "definition", "coding"])


def dataframe_to_markdown(
    frame: pd.DataFrame, digits: int = 4, max_rows: int | None = None
) -> str:
    """tabulate 의존성 없이 작은 DataFrame을 Markdown 표로 변환한다."""
    data = frame.head(max_rows).copy() if max_rows else frame.copy()
    columns = [str(column) for column in data.columns]

    def display(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    raw: pd.DataFrame,
    analysis: pd.DataFrame,
    model_results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    fishers: pd.DataFrame,
    output_dir: Path,
) -> None:
    eligible = analysis[analysis["analysis_eligible"].eq(1)]
    broad = analysis[analysis["analysis_eligible_broad"].eq(1)]
    lawyer_rates = (
        eligible.groupby("lawyer", observed=True)["outcome_suspended"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    comparable = analysis[analysis["below_basic"].notna()]
    fitted = diagnostics[diagnostics["status"].eq("fitted")]
    significant = (
        model_results[
            model_results["significant_05"]
            & model_results["term"].ne("Intercept")
        ]
        if not model_results.empty
        else pd.DataFrame()
    )
    lines = [
        "# 교통범죄(음주운전·도주치상) 탐색적 회귀분석",
        "",
        "## 핵심 결과",
        "",
        f"- 원자료 {len(raw):,}건 중 교통범죄 유죄는 "
        f"{int(((raw['tag_crime_group_manual'] == 'traffic') & (raw['guilty'] == 'Guilty')).sum()):,}건이다.",
        f"- 배정 죄명 후보를 중복 제거하면 {len(analysis):,}건이며, "
        f"변호사 유형과 형종이 확인된 광의 표본은 {len(broad):,}건, "
        f"중대 병합죄·복수피고인·순수 법리사건을 제외한 주 분석 표본은 "
        f"{len(eligible):,}건이다.",
        f"- 사건 시점에 맞는 양형기준과 선고형을 직접 비교할 수 있는 사건은 "
        f"{len(comparable):,}건뿐이다.",
        "- 따라서 회귀계수는 확증적 인과효과가 아니라 표본 내 조건부 연관성으로 해석해야 한다.",
        "",
        "## 변호사 유형별 집행유예 기술통계",
        "",
        dataframe_to_markdown(lawyer_rates),
        "",
        "## 회귀 실행 상태",
        "",
        dataframe_to_markdown(diagnostics),
        "",
    ]
    if not significant.empty:
        lines.extend(
            [
                "## p<0.05 결과",
                "",
                dataframe_to_markdown(
                    significant[
                        [
                            "model",
                            "term",
                            "OR",
                            "CI_lower",
                            "CI_upper",
                            "p_value",
                        ]
                    ]
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## p<0.05 결과",
                "",
                "- 절편을 제외하고 5% 수준에서 유의한 계수는 확인되지 않았다.",
                "",
            ]
        )
    if not fishers.empty:
        lines.extend(
            [
                "## 변호사 유형 쌍별 Fisher 정확검정",
                "",
                dataframe_to_markdown(fishers),
                "",
            ]
        )
    lines.extend(
        [
            "## 해석상 주의",
            "",
            "- 단순 음주운전 양형기준은 2023-07-01 시행분부터 적용된다. 이전 사건에 현재 기준을 소급하지 않았다.",
            "- 양형기준은 원칙적으로 공소제기 시점 기준이나 원자료에는 정확한 공소제기일이 없다. 사건번호의 연도를 대용치로 사용했다.",
            "- 병합죄·복수 피고인 사건에서는 판결문 첫 선고형이 배정 죄명의 독립 형량이 아닐 수 있다. `review_reasons`와 원문 근거를 함께 확인해야 한다.",
            "- 키워드 태깅은 양형이유 표제가 있는 구간을 우선 사용했지만, 표제가 없는 판결문은 전체 본문을 사용하므로 위양성 가능성이 있다.",
            "",
            "## 공식 기준 출처",
            "",
        ]
    )
    for label, url in OFFICIAL_SOURCES.items():
        lines.append(f"- {label}: {url}")
    (output_dir / "traffic_regression_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, low_memory=False)
    analysis, duplicates = build_analysis_data(raw)
    model_results, diagnostics, vifs, fishers = run_models(analysis)
    descriptives = make_descriptives(analysis)
    codebook = codebook_frame()

    audit_columns = [
        "assignment_id",
        "precedent_id",
        "case_number",
        "decision_date_iso",
        "court_name",
        "tag_court_instance_manual",
        "tag_document_focus",
        "tag_sentence_source",
        "case_name",
        "crime_subgroup",
        "drunk_overlap",
        "lawyer",
        "punishment",
        "docket_year",
        "bac",
        "bac_all",
        "bac_type",
        "injury_weeks_max",
        "abandonment",
        "sentence_kind",
        "sentence_months",
        "fine_won",
        "sentence_region",
        "sentence_score",
        "sentence_match",
        "guideline_version",
        "guideline_type",
        "guideline_applicable",
        "basic_low_months",
        "basic_high_months",
        "below_basic",
        "outcome_han",
        "outcome_suspended",
        "outcome_lenient_expanded",
        "settlement",
        "damage_recovery",
        "remorse",
        "no_prior",
        "same_prior",
        "concealment",
        "high_road_risk",
        "settlement_raw",
        "damage_recovery_raw",
        "remorse_raw",
        "no_prior_raw",
        "same_prior_raw",
        "concealment_raw",
        "high_road_risk_raw",
        "factor_scope",
        "factor_evidence_reliable",
        "multi_charge",
        "multi_defendant",
        "serious_other_charge",
        "analysis_eligible_broad",
        "analysis_eligible",
        "manual_review_required",
        "review_reasons",
        "sentence_excerpt",
        "bac_excerpt",
        "injury_excerpt",
        "settlement_excerpt",
        "damage_recovery_excerpt",
        "remorse_excerpt",
        "same_prior_excerpt",
    ]
    analysis[audit_columns].to_csv(
        output_dir / "traffic_candidate_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analysis.to_csv(
        output_dir / "New_anal_traffic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    duplicates.to_csv(
        output_dir / "traffic_duplicate_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_results.to_csv(
        output_dir / "traffic_model_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        output_dir / "traffic_model_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vifs.to_csv(
        output_dir / "traffic_vif.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fishers.to_csv(
        output_dir / "traffic_fisher_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    descriptives.to_csv(
        output_dir / "traffic_descriptives.csv",
        index=False,
        encoding="utf-8-sig",
    )
    codebook.to_csv(
        output_dir / "traffic_codebook.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [{"source": key, "url": value} for key, value in OFFICIAL_SOURCES.items()]
    ).to_csv(
        output_dir / "traffic_official_sources.csv",
        index=False,
        encoding="utf-8-sig",
    )
    create_figures(raw, analysis, model_results, output_dir)
    write_report(
        raw, analysis, model_results, diagnostics, fishers, output_dir
    )

    summary = {
        "raw_n": len(raw),
        "traffic_guilty_n": int(
            (
                raw["tag_crime_group_manual"].eq("traffic")
                & raw["guilty"].eq("Guilty")
            ).sum()
        ),
        "target_unique_n": len(analysis),
        "analysis_eligible_broad_n": int(
            analysis["analysis_eligible_broad"].sum()
        ),
        "analysis_eligible_n": int(analysis["analysis_eligible"].sum()),
        "guideline_comparable_n": int(analysis["below_basic"].notna().sum()),
        "manual_review_required_n": int(analysis["manual_review_required"].sum()),
        "model_fitted_n": int(diagnostics["status"].eq("fitted").sum()),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
