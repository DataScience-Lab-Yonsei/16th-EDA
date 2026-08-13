#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py — 관광데이터 × 정규화 기상·대기질 마스터 CSV 생성
키: (signguCode, date). 관광(현지인/외지인/외국인/합계) + 기상 S_* + 대기질 S_*.
- 기상: ASOS+AWS를 시군구 기초단위(광역시는 시도풀)로 배정, 지점 평균.
- 대기질: AirKorea 측정소를 동일 규칙으로 배정, 평균.
- 광역시 자치구는 해당 시(도) 평균을 공유(도시규모 기상/대기질).
"""
import pandas as pd, numpy as np, re, sys

TOUR="/mnt/user-data/uploads/Desktop/tour_visitor_locgo_sigungu_daily_2023_2025_merged.csv"
META="/root/.claude/uploads/c1b7af1e-dfdd-5d68-832d-7596ebedd996/e2e066a9-META________20260716163938.csv"
AK_KEYSRC="/tmp/akpeek/2026#Ub144 1#Uc6d4.xlsx"
BASE="/home/claude/normalized_2023_2026/"

SIDO_PREFIX=[('서울','11'),('부산','26'),('대구','27'),('인천','28'),('광주광역','29'),('대전','30'),
 ('울산','31'),('세종','36'),('경기','41'),('충청북','43'),('충북','43'),('충청남','44'),('충남','44'),
 ('전라남','46'),('전남','46'),('경상북','47'),('경북','47'),('경상남','48'),('경남','48'),
 ('제주','50'),('강원','51'),('전라북','52'),('전북','52')]
METRO={'11','26','27','28','29','30','31','36'}

def sido_of(prov):
    for k,v in SIDO_PREFIX:
        if prov.startswith(k): return v
    return None
def base_of(rest):
    m=re.search(r'([가-힣]+시)',rest) or re.search(r'([가-힣]+군)',rest) or re.search(r'([가-힣]+구)',rest)
    return m.group(1) if m else None
def key_from_addr(addr):
    a=str(addr).replace('(산지)','').strip(); toks=a.split()
    if len(toks)<2: return None
    prov=toks[0]; rest=' '.join(toks[1:]); base=base_of(rest)
    if '광주통합' in prov or prov.startswith('전남광주'):
        sido='29' if (base and base.endswith('구')) else '46'
    else:
        sido=sido_of(prov)
    if sido is None: return None
    return sido if sido in METRO else (f"{sido}|{base}" if base else None)
def key_from_signgu(code,nm):
    sido=str(code).zfill(5)[:2]; base=str(nm).split()[0]
    return sido if sido in METRO else f"{sido}|{base}"

def main():
    # ---- 1. 지점(ASOS/AWS) → key ----
    meta=pd.read_csv(META,encoding='cp949',skiprows=1); meta.columns=[c.strip() for c in meta.columns]
    meta['종료일']=meta['종료일'].fillna('9999-12-31')
    meta=meta.sort_values(['지점','종료일']).groupby('지점',as_index=False).last().set_index('지점')
    def stn_key(s):
        a=meta.loc[s,'지점주소'] if s in meta.index else None
        return key_from_addr(a) if isinstance(a,str) else None

    # ---- 2. 기상 정규화 로드 + key ----
    met=[]
    for f in ['asos_2023_2025_normalized.csv','aws_2023_2025_normalized.csv']:
        d=pd.read_csv(BASE+f, usecols=['station','date','S_temp','S_humidity','S_wind','S_precip'])
        met.append(d)
    met=pd.concat(met,ignore_index=True)
    met['key']=met['station'].map(stn_key)
    met=met.dropna(subset=['key'])
    wx=met.groupby(['key','date'])[['S_temp','S_humidity','S_wind','S_precip']].mean().reset_index()
    print("기상 (key,date):",len(wx))

    # ---- 3. 대기질(AirKorea) → key ----
    aksrc=pd.read_excel(AK_KEYSRC,engine="calamine",usecols=["측정소코드","주소"]).dropna().drop_duplicates("측정소코드")
    aksrc['key']=aksrc['주소'].apply(key_from_addr)
    code2key=dict(zip(aksrc['측정소코드'],aksrc['key']))
    ak=pd.read_csv(BASE+'airkorea_2023_2025_normalized.csv',usecols=['station','date','S_airquality','S_aq_mean'])
    ak['key']=ak['station'].map(code2key); ak=ak.dropna(subset=['key'])
    aq=ak.groupby(['key','date'])[['S_airquality','S_aq_mean']].mean().reset_index()
    print("대기질 (key,date):",len(aq))

    # ---- 4. 관광 피벗 ----
    t=pd.read_csv(TOUR,encoding='utf-8-sig')
    t['date']=pd.to_datetime(t['baseYmd'],format='%Y%m%d').dt.strftime('%Y-%m-%d')
    piv=t.pivot_table(index=['signguCode','signguNm','date'],columns='touDivNm',values='touNum',aggfunc='sum').reset_index()
    piv.columns.name=None
    ren={'현지인(a)':'visitor_local','외지인(b)':'visitor_domestic','외국인(c)':'visitor_foreign'}
    piv=piv.rename(columns=ren)
    vcols=[c for c in ['visitor_local','visitor_domestic','visitor_foreign'] if c in piv.columns]
    piv['visitor_total']=piv[vcols].sum(axis=1)
    piv['key']=[key_from_signgu(c,n) for c,n in zip(piv['signguCode'],piv['signguNm'])]
    print("관광 (시군구,date):",len(piv))

    # ---- 5. 조인 ----
    m=piv.merge(wx,on=['key','date'],how='left').merge(aq,on=['key','date'],how='left')
    cols=['signguCode','signguNm','date','visitor_local','visitor_domestic','visitor_foreign','visitor_total',
          'S_temp','S_humidity','S_wind','S_precip','S_airquality','S_aq_mean']
    cols=[c for c in cols if c in m.columns]
    m=m[cols].sort_values(['signguCode','date'])
    m.to_csv('/home/claude/master_tourism_weather_2023_2025.csv',index=False,encoding='utf-8-sig')
    # 커버리지 리포트
    print("\n=== 마스터 ===")
    print("행:",len(m),"| 시군구:",m['signguCode'].nunique(),"| 기간:",m['date'].min(),"~",m['date'].max())
    for c in ['S_temp','S_humidity','S_wind','S_precip','S_airquality']:
        print(f"  {c:14s} 결측 {m[c].isna().mean()*100:4.1f}%")
    # 완전 결측 시군구
    wxmiss=m.groupby('signguNm')['S_temp'].apply(lambda s:s.isna().all())
    aqmiss=m.groupby('signguNm')['S_airquality'].apply(lambda s:s.isna().all())
    print("기상 전무 시군구:", list(wxmiss[wxmiss].index))
    print("대기질 전무 시군구 수:", int(aqmiss.sum()), "예:", list(aqmiss[aqmiss].index)[:15])

if __name__=="__main__": main()
