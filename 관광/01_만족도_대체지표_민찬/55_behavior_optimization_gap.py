"""55 — '최적화 추천을 따른 여행' vs '그냥 간 여행'의 실제 행동지표 격차.
목적 재정의: 만족도(잘 안 움직임, 54번 확인) 대신 소비·재방문의향·실망회피(경고구간)를 핵심 지표로.
이행도 점수 4개(순환성 있는 '마무리' 항목 제외): 동기스윗스팟/워스트조합회피/베스트조합포함/장소검증."""
import route_common as rc, pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic'); plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.join(rc.BASE, "plots_deep"); os.makedirs(OUT, exist_ok=True)

d = rc.build(); vf = rc.fine_category(d['visit']); tdf = d['tdf'].copy()
vf['SAT'] = pd.to_numeric(vf['DGSTFN'], errors='coerce')

BEST = {'자연 탐방형': {'전통시장', '레저/스포츠 관련 시설(스키, 카트, 수상레저)'},
        '도심 쇼핑·문화형': {'레저/스포츠 관련 시설(스키, 카트, 수상레저)', '카페(목적·핫플)'},
        '역사·유적 탐방형': {'맛집(원정)', '테마시설(놀이공원, 워터파크)'},
        '테마파크·가족형': {'쇼핑몰·아울렛', '상점'},
        '레저·액티비티형': {'산책로, 둘레길 등', '체험 활동 관광지'},
        '축제·이벤트형': {'맛집(원정)', '테마시설(놀이공원, 워터파크)'}}
WORST = {'자연 탐방형': {'지역 축제/행사'}, '도심 쇼핑·문화형': {'전통시장'},
         '역사·유적 탐방형': {'쇼핑몰·아울렛'}, '테마파크·가족형': {'전통시장'},
         '레저·액티비티형': {'역사/유적/종교 시설(문화재, 박물관, 촬영지, 절 등)'}, '축제·이벤트형': {'카페(목적·핫플)'}}
pc = pd.read_csv(os.path.join(rc.BASE, "place_crowd.csv")).set_index('NM')
GOOD_TIER = set(pc[pc['유형'].isin(['대표명소(붐빔+만족)', '숨은명소(한적+만족)'])].index)
BAD_TIER = set(pc[pc['유형'] == '과대평가(붐빔+불만)'].index)

cl_map = dict(zip(tdf['TRAVEL_ID'], tdf['CL_NM']))
seek = vf.assign(x=(vf['INTENT'] == '목적형')).groupby('TRAVEL_ID')['x'].mean()
tour = vf[vf['FINE_CAT'].isin(rc.FINE_TOUR)].copy()
tour['CL_NM'] = tour['TRAVEL_ID'].map(cl_map)

rows = []
for tid, g in tour.groupby('TRAVEL_ID'):
    cl = g['CL_NM'].iloc[0]
    cats = set(g['FINE_CAT'])
    sk = seek.get(tid, np.nan)
    c1 = int(0.4 <= sk <= 0.8) if pd.notna(sk) else 0
    c2 = int(len(cats & WORST.get(cl, set())) == 0)
    c3 = int(len(cats & BEST.get(cl, set())) > 0)
    nms = set(g['VISIT_AREA_NM'].dropna().astype(str))
    c4 = int(len(nms & BAD_TIER) == 0 and len(nms & GOOD_TIER) > 0)
    rows.append((tid, c1, c2, c3, c4))
sdf = pd.DataFrame(rows, columns=['TRAVEL_ID', 'c1', 'c2', 'c3', 'c4'])
sdf['SCORE'] = sdf[['c1', 'c2', 'c3', 'c4']].sum(axis=1)
tdf = tdf.merge(sdf, on='TRAVEL_ID', how='left')
tdf['그룹'] = pd.cut(tdf['SCORE'], [-1, 1, 2, 4], labels=['하위(0-1)', '보통(2)', '최적화(3-4)'])

order = ['하위(0-1)', '보통(2)', '최적화(3-4)']
summ = tdf.groupby('그룹', observed=True).agg(
    N=('TRAVEL_ID', 'count'), 소비=('SPEND', 'median'), 재방문의향=('REVISIT', 'mean'),
    경고구간비율=('SAT', lambda s: (s <= 3).mean() * 100), 방문지수=('N_VISITS', 'median')).reindex(order)
print("=== 이행도 그룹별 실제 행동지표 (전체평균 = 참고선) ===")
print(summ.round(3).to_string())
print(f"\n전체 평균 소비: {tdf['SPEND'].median():,.0f}원 | 전체 평균 재방문의향: {tdf['REVISIT'].mean():.3f} | 전체 경고비율: {(tdf['SAT']<=3).mean()*100:.2f}%")

spend_gap = summ.loc['최적화(3-4)', '소비'] / summ.loc['하위(0-1)', '소비']
rev_gap = summ.loc['최적화(3-4)', '재방문의향'] - summ.loc['하위(0-1)', '재방문의향']
warn_gap_abs = summ.loc['하위(0-1)', '경고구간비율'] - summ.loc['최적화(3-4)', '경고구간비율']
print(f"\n소비 배율(최적화/하위): {spend_gap:.2f}배")
print(f"재방문의향 격차: {rev_gap:+.3f}")
print(f"경고구간 절대감소: {warn_gap_abs:.2f}%p")

# 요소 하나씩만 지켰을 때 개별 기여도 (어떤 걸 최적화하는게 제일 큰가)
print("\n=== 개별 요소 하나만 지켰을 때 vs 안 지켰을 때 (소비 중앙값) ===")
for c, label in [('c1', '동기스윗스팟'), ('c2', '워스트조합회피'), ('c3', '베스트조합포함'), ('c4', '장소검증')]:
    with_ = tdf[tdf[c] == 1]['SPEND'].median()
    without = tdf[tdf[c] == 0]['SPEND'].median()
    print(f"  {label}: 지킴 {with_:,.0f}원 vs 안지킴 {without:,.0f}원 (배율 {with_/without:.2f})")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
colors = ['#c44e52', '#8c8c8c', '#4c72b0']
ax = axes[0]
bars = ax.bar(order, summ['소비'] / 10000, color=colors)
for b, v in zip(bars, summ['소비']): ax.text(b.get_x()+b.get_width()/2, v/10000+0.5, f'{v/10000:.1f}만', ha='center')
ax.set_title(f'이행도별 소비(중앙값)\n최적화/하위 = {spend_gap:.2f}배')
ax = axes[1]
bars = ax.bar(order, summ['재방문의향'], color=colors)
for b, v in zip(bars, summ['재방문의향']): ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center')
ax.set_ylim(3.9, 4.3); ax.set_title(f'이행도별 재방문의향\n격차 {rev_gap:+.3f}')
ax = axes[2]
bars = ax.bar(order, summ['경고구간비율'], color=colors)
for b, v in zip(bars, summ['경고구간비율']): ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.2f}%', ha='center')
ax.set_title(f'이행도별 경고구간(만족≤3) 비율\n절대감소 {warn_gap_abs:.2f}%p')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "55_behavior_optimization_gap.png"), dpi=120)
plt.close()
print(f"\n완료 → {OUT}/55_behavior_optimization_gap.png")
