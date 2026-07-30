#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_hidden_spot.py
=================
계절별 '히든스팟' 도출.

정의 : 그 계절에 (1) 안 붐비고(혼잡도 낮음) (2) 날씨가 좋고(KTCI ≥ 그 계절 중앙값)
       (3) 볼거리가 있는(매력도 높음) 시군구.

점수 산출
    혼잡도  crowd   = z( log(방문객수) )                 # 낮을수록 여유
    매력도  appeal  = z(관광객다양성) + z(관광소비강도) + z(타권역방문자비중)   # 세 지표 동일 가중
    히든점수 hidden = appeal − crowd                     # 좋은 날씨 통과 지역 중 상위

핵심 : 단위가 다른 지표를 표준화(z)로 같은 잣대에 맞춘 뒤, 계절마다 따로 순위화한다.

입력  : master_monthly_ktci_tourism.csv   (날씨×관광 월별 마스터)
출력  : hidden_by_season.csv              (계절별 상위 후보)

의존성 : pandas, numpy
사용법 : python 03_hidden_spot.py --master ../Dataset/master_monthly_ktci_tourism.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

SEASONS = {12: "겨울", 1: "겨울", 2: "겨울",
           3: "봄", 4: "봄", 5: "봄",
           6: "여름", 7: "여름", 8: "여름",
           9: "가을", 10: "가을", 11: "가을"}


def zscore(s: pd.Series) -> pd.Series:
    """표준화 (평균 0, 표준편차 1). 표준편차 0이면 0으로."""
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def build_hidden(df: pd.DataFrame, ktci_col: str = "KTCI_hybrid", top_n: int = 12) -> pd.DataFrame:
    """월별 마스터 → 계절별 히든스팟 순위표."""
    df = df.copy()
    df["month"] = df["baseYm"].astype(str).str[-2:].astype(int)
    df["season"] = df["month"].map(SEASONS)

    # 시군구×계절 평균 (여러 달을 계절로 집계)
    agg = (df.groupby(["season", "signguCode", "signguNm"])
             .agg(KTCI=(ktci_col, "mean"),
                  visitor=("visitor_total", "mean"),
                  diversity=("관광객다양성", "mean"),
                  spend=("관광소비강도", "mean"),
                  outside=("타권역방문자비중", "mean"))
             .reset_index())

    out = []
    for season, g in agg.groupby("season"):
        g = g.dropna(subset=["KTCI", "visitor", "diversity", "spend", "outside"]).copy()
        # (2) 좋은 날씨 필터 : 그 계절 KTCI 중앙값 이상
        g = g[g["KTCI"] >= g["KTCI"].median()]
        # (1) 안 붐빔 필터 : 방문객 중앙값 이하(비혼잡군)만 후보로
        g["crowd"] = zscore(np.log1p(g["visitor"]))
        g = g[g["visitor"] <= g["visitor"].median()]
        # (3) 매력도 : 비혼잡군 안에서 표준화해 합산, 매력에서 혼잡을 빼 최종 점수
        g["appeal"] = zscore(g["diversity"]) + zscore(g["spend"]) + zscore(g["outside"])
        g["hidden_score"] = g["appeal"] - zscore(np.log1p(g["visitor"]))
        g = g.sort_values("hidden_score", ascending=False).head(top_n)
        out.append(g)

    res = pd.concat(out, ignore_index=True)
    order = {"봄": 0, "여름": 1, "가을": 2, "겨울": 3}
    return res.sort_values(["season", "hidden_score"],
                           key=lambda c: c.map(order) if c.name == "season" else c,
                           ascending=[True, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="../Dataset/master_monthly_ktci_tourism.csv")
    ap.add_argument("--ktci", default="KTCI_hybrid",
                    help="사용할 KTCI 컬럼 (KTCI_data / KTCI_2014_adapted / KTCI_hybrid)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="hidden_by_season.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    res = build_hidden(df, ktci_col=args.ktci, top_n=args.top)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] {args.out} 저장 ({len(res)}행)")
    for season in ["봄", "여름", "가을", "겨울"]:
        top = res[res.season == season].head(1)
        if len(top):
            r = top.iloc[0]
            print(f"  {season} 1위 : {r.signguNm}  (KTCI {r.KTCI:.1f}, 히든점수 {r.hidden_score:.2f})")


if __name__ == "__main__":
    main()
