#!/usr/bin/env python3
"""
성범죄 전처리·태깅

(1) 성범죄 + 유죄 + 하급심
(2) sex_type · 양형 성격 플래그
(3) 형량 year
(4) 전치 주수 injury_weeks
(5) 감/가중요소 태깅
(6) 양형 단순구간 비교 → reduced
(7) outcome → data/New_anal_sex.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, NEW_ANAL_SEX, OUTPUT, find_preprocessed


# ---------------------------------------------------------------------------
# keyword helpers
# ---------------------------------------------------------------------------

def kw_pos(text: str, patterns: list[str], neg_window: int = 18) -> bool:
    if not isinstance(text, str) or not text:
        return False
    neg = re.compile(r"(?:않|못|없|아니|결렬|불발|거부)")
    for p in patterns:
        for m in re.finditer(p, text):
            tail = text[m.end() : m.end() + neg_window]
            head = text[max(0, m.start() - 8) : m.start()]
            blob = head + m.group(0) + tail
            if neg.search(tail) or re.search(r"이르지\s*못|되지\s*않|하지\s*않", blob):
                continue
            after = text[m.end() : m.end() + 1]
            if "합의" in p and after in ("체", "부"):
                continue
            return True
    return False


def classify_sex_type(name: str, group: str) -> str | None:
    t = f"{group} {name}"
    if any(x in t for x in ["성매매", "음란물제작", "성착취물", "통신매체이용음란", "음란물유포"]):
        if not any(x in t for x in ["촬영", "카메라", "촬영물"]):
            return None
    if any(x in t for x in ["촬영", "카메라", "불법촬영", "촬영물"]):
        return "camera"
    if any(x in t for x in ["강간", "유사강간", "미성년자의제강간"]):
        return "rape"
    if any(x in t for x in ["강제추행", "준강제추행", "추행"]):
        return "indecent"
    return None


def guideline_flags(text: str, sex_type: str) -> dict:
    t = text or ""
    return {
        "minor_victim": int(bool(re.search(r"미성년|아동|청소년|13세\s*미만|16세\s*미만", t))),
        "relative": int(bool(re.search(r"친족|혈족|인척|의제친족|보호자", t))),
        "special": int(bool(re.search(r"흉기|위력|심신상실|항거불능|특수강제|특수강간|집단", t))),
        "distribute": int(bool(re.search(r"유포|게시|전송|배포|촬영물등이용", t)))
        if sex_type == "camera"
        else 0,
        "habit_flag": int(bool(re.search(r"상습|누범|반복적\s*범행", t))),
    }


def extract_sentence_years(text: str) -> float | None:
    if not isinstance(text, str) or not text:
        return None
    order = text
    m_ord = re.search(r"【\s*주\s*문\s*】(.{0,800})", text)
    if m_ord:
        order = m_ord.group(1)
    for i, pat in enumerate(
        [r"징역\s*(\d+)\s*년\s*(\d+)\s*월", r"징역\s*(\d+)\s*년", r"징역\s*(\d+)\s*개?월"]
    ):
        m = re.search(pat, order) or re.search(pat, text)
        if not m:
            continue
        if i == 0:
            return float(m.group(1)) + float(m.group(2)) / 12.0
        if i == 1:
            return float(m.group(1))
        return float(m.group(1)) / 12.0
    return None


def extract_injury_weeks(text: str) -> float | None:
    if not isinstance(text, str):
        return None
    m = re.search(r"전치\s*(\d+)\s*주", text) or re.search(r"(\d+)\s*주\s*의\s*치료", text)
    return float(m.group(1)) if m else None


# 연구용 단순 양형 구간 (연)
GUIDELINE = {
    ("indecent", 0, 0): (0.5, 2.0),
    ("indecent", 0, 1): (1.0, 3.0),
    ("indecent", 1, 0): (1.0, 3.0),
    ("indecent", 1, 1): (1.5, 4.0),
    ("rape", 0, 0): (2.5, 5.0),
    ("rape", 0, 1): (3.0, 7.0),
    ("rape", 1, 0): (3.0, 7.0),
    ("rape", 1, 1): (4.0, 8.0),
    ("camera", 0, 0): (0.5, 1.5),
    ("camera", 0, 1): (1.0, 3.0),
    ("camera", 1, 0): (1.0, 2.5),
    ("camera", 1, 1): (1.5, 4.0),
}


def judge_reduced(sex_type: str, year: float | None, flags: dict) -> str | None:
    if year is None or sex_type not in ("indecent", "rape", "camera"):
        return None
    special = int(
        flags.get("special", 0) or flags.get("relative", 0) or flags.get("distribute", 0)
    )
    minor = int(flags.get("minor_victim", 0))
    low, high = GUIDELINE[(sex_type, minor, 1 if special else 0)]
    if year < low:
        return "reduced"
    if year > high:
        return "aggravated"
    return "normal"


def tag_factors(text: str) -> dict:
    agree = kw_pos(
        text,
        [r"합의(?!체|부)", r"처벌불원", r"처벌을\s*원하지\s*않", r"합의금", r"원만히\s*합의"],
    )
    reflection = kw_pos(text, [r"반성", r"뉘우치"])
    first = kw_pos(text, [r"초범", r"전과가?\s*없", r"처벌받은\s*전력이?\s*없"])
    same = False
    if isinstance(text, str):
        for m in re.finditer(
            r"동종\s*(?:전과|범죄|범행)|성범죄\s*전과|강제추행\s*전과|강간\s*전과", text
        ):
            if re.search(r"없|아니", text[m.start() : m.end() + 15]):
                continue
            same = True
            break
    recovery = kw_pos(
        text, [r"위자료", r"합의금\s*지급", r"공탁", r"피해(?:를|가)?\s*회복", r"손해배상"]
    )
    conceal = kw_pos(text, [r"은닉", r"은폐", r"증거인멸", r"증거를\s*인멸"])
    deny = kw_pos(text, [r"범행을\s*부인", r"공소사실을\s*부인", r"허위\s*진술"])
    return {
        "agree": int(agree),
        "reflection": int(reflection),
        "first_offender": int(first),
        "same_record": int(same),
        "recovery": int(recovery),
        "conceal": int(conceal),
        "deny": int(deny),
    }


def main() -> None:
    src = find_preprocessed()
    print(f"[01] load: {src}")
    df = pd.read_csv(src, low_memory=False)
    df.columns = [c.lstrip("\ufeff").strip('"') for c in df.columns]

    # --- (1) 성범죄 + 유죄 + 하급심 ---
    g = df["tag_crime_group_manual"].fillna("").astype(str)
    n = df["tag_crime_name_manual"].fillna("").astype(str)
    mask = g.str.contains("sex|성범죄", case=False) | n.str.contains(
        "강제추행|강간|촬영|성폭력|준강제|추행|카메라", na=False
    )
    sex = df[mask].copy()
    sex["sex_type"] = [
        classify_sex_type(a, b)
        for a, b in zip(
            sex["tag_crime_name_manual"].fillna(""),
            sex["tag_crime_group_manual"].fillna(""),
        )
    ]
    core = sex[sex["sex_type"].isin(["indecent", "rape", "camera"])].copy()
    core = core[core["guilty"] == "Guilty"]
    core = core[~core["court_name"].fillna("").str.contains("대법원")].reset_index(drop=True)
    print(f"[01] guilty lower core n={len(core)}")
    print(core["sex_type"].value_counts().to_string())

    rows = []
    for _, r in core.iterrows():
        text = r.get("full_text") or ""
        st = r["sex_type"]
        flags = guideline_flags(text, st)
        factors = tag_factors(text)
        year = extract_sentence_years(text)
        reduced = judge_reduced(st, year, flags)
        pun = r.get("punishment")
        if pun == "suspended" or reduced == "reduced":
            outcome = True
        else:
            outcome = False

        rows.append(
            {
                "precedent_id": r.get("precedent_id"),
                "case_number": r.get("case_number"),
                "case_name": r.get("case_name"),
                "court_name": r.get("court_name"),
                "decision_date_iso": r.get("decision_date_iso"),
                "crime_name": r.get("tag_crime_name_manual"),
                "sex_type": st,
                "lawyer": r.get("lawyer"),
                "punishment": pun,
                "guilty": r.get("guilty"),
                "year": year,
                "injury_weeks": extract_injury_weeks(text),
                "reduced": reduced,
                "outcome": outcome,
                **flags,
                **factors,
            }
        )

    out = pd.DataFrame(rows)
    NEW_ANAL_SEX.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(NEW_ANAL_SEX, index=False, encoding="utf-8-sig")

    # 간단 기술통계
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out.groupby("sex_type")["outcome"].agg(["count", "mean"]).to_csv(
        OUTPUT / "01_desc_sextype.csv", encoding="utf-8-sig"
    )
    out.dropna(subset=["lawyer"]).groupby("lawyer")["outcome"].agg(["count", "mean"]).to_csv(
        OUTPUT / "01_desc_lawyer.csv", encoding="utf-8-sig"
    )

    print(f"[01] saved {NEW_ANAL_SEX}  rows={len(out)}")
    print(f"     outcome rate={out['outcome'].mean():.1%}")
    print(f"     lawyer NA={out['lawyer'].isna().sum()}")
    print("[01] done — next: python3 code/02_regression.py")


if __name__ == "__main__":
    main()
