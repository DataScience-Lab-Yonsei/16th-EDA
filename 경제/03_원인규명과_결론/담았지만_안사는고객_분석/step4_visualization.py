"""
Step 4. 시각화 (Visualization)
================================
분석 결과를 6개의 차트 이미지로 저장합니다.

출력 파일:
  - fig1_4types.png          : 4개 고객 유형 분류 막대그래프
  - fig2_comparison.png      : 핵심 지표 비교 (담고 삼 / 안 삼 / 진성 잠재 고객)
  - fig3_tenure_breakdown.png: 활동 기간별 세분화 (3개 서브차트)
  - fig4_serious_profile.png : 진성 잠재 고객 상세 프로파일
  - fig5_tenure_dist.png     : 활동 기간 분포 비교 (비율 + 유저 수)
  - fig6_products_dist.png   : 담은 고유 상품 수 분포 비교

사전 조건:
  - step2_classify_users.py 실행 후 prof_bought.csv, prof_cnb.csv 생성 필요

실행 방법:
    python step4_visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# ── 한국어 폰트 설정 ───────────────────────────────────────
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fp = fm.FontProperties(fname=FONT_PATH)
plt.rcParams['font.family'] = fp.get_name()
plt.rcParams['axes.unicode_minus'] = False

BC = '#1565C0'   # 담고 삼 (파랑)
RC = '#C62828'   # 담았지만 안 삼 (빨강)
AC = '#2E7D32'   # 진성 잠재 고객 (초록)

print("=" * 60)
print("Step 4. 시각화")
print("=" * 60)

print("\n데이터 로딩 중...")
prof_bought = pd.read_csv('prof_bought.csv')
prof_cnb    = pd.read_csv('prof_cnb.csv')

bins_t   = [-0.001, 1, 7, 30, 60, 145]
labels_t = ['1일\n이하', '1~7일', '7~30일', '30~60일', '60일\n초과']
bins_p   = [0, 1, 3, 10, 50, 2200]
labels_p = ['1개', '2~3개', '4~10개', '11~50개', '51개+']

prof_cnb['tenure_bin']    = pd.cut(prof_cnb['u_tenure_d'],    bins=bins_t, labels=labels_t)
prof_cnb['prod_bin']      = pd.cut(prof_cnb['n_products'],     bins=bins_p, labels=labels_p)
prof_bought['tenure_bin'] = pd.cut(prof_bought['u_tenure_d'],  bins=bins_t, labels=labels_t)
prof_bought['prod_bin']   = pd.cut(prof_bought['n_products'],   bins=bins_p, labels=labels_p)

active_serious = prof_cnb[(prof_cnb['u_tenure_d'] >= 7) & (prof_cnb['n_products'] >= 4)]

# ── Fig 1: 4개 고객 유형 분류 ──────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
types  = ['담지도\n사지도 않음', '담았지만\n안 삼', '담고 삼', '안 담고\n구매']
counts = [247566, len(prof_cnb), len(prof_bought), 183]
colors = ['#90A4AE', RC, BC, '#FFA000']
bars = ax.bar(types, counts, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
ax.set_title('4개 고객 유형 분류 (Notion 원본 기준 + CSV 실측)', fontproperties=fp, fontsize=14, fontweight='bold')
ax.set_ylabel('유저 수', fontproperties=fp)
for bar, cnt in zip(bars, counts):
    pct = cnt / sum(counts) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
            f'{cnt:,}명\n({pct:.1f}%)', ha='center', fontproperties=fp, fontsize=10.5)
ax.set_ylim(0, max(counts) * 1.2)
ax.text(0.01, 0.02,
        '* "담지도 사지도 않음"(247,566명)과 "안 담고 구매"(183명)는 Notion 원본 로그 기준\n'
        '  현재 CSV는 cart 이벤트 기준 전처리 데이터이므로 이 두 유형은 포함되지 않음',
        transform=ax.transAxes, fontproperties=fp, fontsize=8.5, color='gray', va='bottom')
plt.tight_layout()
plt.savefig('fig1_4types.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig1_4types.png 저장 완료")

# ── Fig 2: 핵심 지표 비교 ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('고객 유형별 핵심 지표 비교\n(파랑=담고 삼 / 빨강=담았지만 안 삼 / 초록=진성 잠재 고객)',
             fontproperties=fp, fontsize=14, fontweight='bold')

def triple_bar(ax, xlabels, b_vals, c_vals, a_vals, title, ylabel):
    x = np.arange(len(xlabels)); w = 0.26
    ax.bar(x - w, b_vals, w, label='담고 삼',         color=BC, alpha=0.85)
    ax.bar(x,     c_vals, w, label='담았지만 안 삼',   color=RC, alpha=0.85)
    ax.bar(x + w, a_vals, w, label='진성 잠재 고객',   color=AC, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontproperties=fp, fontsize=9)
    ax.set_title(title, fontproperties=fp, fontsize=11)
    ax.set_ylabel(ylabel, fontproperties=fp)
    ax.legend(prop=fp, fontsize=8)

# 활동 기간 분포
cnb_t    = prof_cnb['tenure_bin'].value_counts(normalize=True).sort_index() * 100
bought_t = prof_bought['tenure_bin'].value_counts(normalize=True).sort_index() * 100
act_t    = active_serious['tenure_bin'].value_counts(normalize=True).sort_index() * 100
triple_bar(axes[0,0], labels_t, bought_t.values, cnb_t.values, act_t.values, '활동 기간 분포', '비율 (%)')

# 담은 상품 수 분포
cnb_p    = prof_cnb['prod_bin'].value_counts(normalize=True).sort_index() * 100
bought_p = prof_bought['prod_bin'].value_counts(normalize=True).sort_index() * 100
act_p    = active_serious['prod_bin'].value_counts(normalize=True).sort_index() * 100
triple_bar(axes[0,1], labels_p, bought_p.values, cnb_p.values, act_p.values, '담은 고유 상품 수 분포', '비율 (%)')

# u_prior_cvr 분포
bins_cvr   = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 25]
labels_cvr = ['0~5%', '5~10%', '10~20%', '20~30%', '30~50%', '50%+']
cnb_cvr    = pd.cut(prof_cnb['u_prior_cvr'],       bins=bins_cvr, labels=labels_cvr).value_counts(normalize=True).sort_index() * 100
bought_cvr = pd.cut(prof_bought['u_prior_cvr'],     bins=bins_cvr, labels=labels_cvr).value_counts(normalize=True).sort_index() * 100
act_cvr    = pd.cut(active_serious['u_prior_cvr'],  bins=bins_cvr, labels=labels_cvr).value_counts(normalize=True).sort_index() * 100
triple_bar(axes[0,2], labels_cvr, bought_cvr.values, cnb_cvr.values, act_cvr.values, 'u_prior_cvr 분포', '비율 (%)')

# 삭제 행동 비율 분포 (담았지만 안 삼 내부)
prof_cnb['remove_bin'] = pd.cut(prof_cnb['remove_rate'],
    bins=[-0.001, 0, 0.25, 0.5, 0.75, 1.001],
    labels=['0%\n방치만', '0~25%', '25~50%', '50~75%', '75~100%\n삭제위주'])
remove_dist = prof_cnb['remove_bin'].value_counts(normalize=True).sort_index() * 100
axes[1,0].bar(remove_dist.index, remove_dist.values,
              color=['#FFCCBC','#FF8A65','#FF5722','#E64A19','#BF360C'], edgecolor='white')
axes[1,0].set_title('담았지만 안 삼: 삭제 행동 비율 분포', fontproperties=fp, fontsize=11)
axes[1,0].set_ylabel('비율 (%)', fontproperties=fp)
for bar, val in zip(axes[1,0].patches, remove_dist.values):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f'{val:.1f}%', ha='center', fontproperties=fp, fontsize=9)

# 마지막 이벤트 경과일 분포
bins_d   = [0, 7, 30, 60, 90, 145]
labels_d = ['7일\n이내', '7~30일', '30~60일', '60~90일', '90일\n초과']
cnb_rec   = pd.cut(prof_cnb['u_recency_d'].fillna(999),    bins=bins_d, labels=labels_d).value_counts(normalize=True).sort_index() * 100
bought_rec = pd.cut(prof_bought['u_recency_d'].fillna(999), bins=bins_d, labels=labels_d).value_counts(normalize=True).sort_index() * 100
x = np.arange(len(labels_d)); w2 = 0.35
axes[1,1].bar(x-w2/2, bought_rec.values, w2, label='담고 삼',       color=BC, alpha=0.85)
axes[1,1].bar(x+w2/2, cnb_rec.values,   w2, label='담았지만 안 삼', color=RC, alpha=0.85)
axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(labels_d, fontproperties=fp, fontsize=9)
axes[1,1].set_title('마지막 이벤트 이후 경과일 분포', fontproperties=fp, fontsize=11)
axes[1,1].set_ylabel('비율 (%)', fontproperties=fp)
axes[1,1].legend(prop=fp, fontsize=9)

# 진성 잠재 고객 vs 담고 삼 상대 비율
metrics_label = ['누적\n담기수', '누적\n조회수', '활동\n기간', '담은\n상품수', '세션\n깊이']
metrics_col   = ['u_carts', 'u_views', 'u_tenure_d', 'n_products', 'avg_s_depth']
b_vals = [prof_bought[c].mean() for c in metrics_col]
a_vals = [active_serious[c].mean() for c in metrics_col]
a_norm = [a/b*100 if b > 0 else 0 for a, b in zip(a_vals, b_vals)]
x = np.arange(len(metrics_label)); w3 = 0.35
axes[1,2].bar(x-w3/2, [100]*len(metrics_label), w3, label='담고 삼 (기준=100%)', color=BC, alpha=0.7)
axes[1,2].bar(x+w3/2, a_norm,                   w3, label='진성 잠재 고객',       color=AC, alpha=0.85)
axes[1,2].axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.6)
axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(metrics_label, fontproperties=fp, fontsize=9)
axes[1,2].set_title('진성 잠재 고객 vs 담고 삼\n(담고 삼 = 100% 기준)', fontproperties=fp, fontsize=11)
axes[1,2].set_ylabel('상대 비율 (%)', fontproperties=fp)
axes[1,2].legend(prop=fp, fontsize=9)
for i, norm in enumerate(a_norm):
    axes[1,2].text(i+w3/2, norm+1.5, f'{norm:.0f}%', ha='center', fontproperties=fp, fontsize=8.5)

plt.tight_layout()
plt.savefig('fig2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig2_comparison.png 저장 완료")

# ── Fig 3: 활동 기간별 세분화 ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('"담았지만 안 삼" 내부 세분화: 활동 기간별 행동 패턴',
             fontproperties=fp, fontsize=14, fontweight='bold')

tenure_agg = prof_cnb.groupby('tenure_bin', observed=True).agg(
    유저수=('user_hash', 'count'),
    평균_carts=('u_carts', 'mean'),
    평균_views=('u_views', 'mean'),
    평균_removes=('u_removes', 'mean'),
    평균_상품수=('n_products', 'mean'),
    평균_prior_cvr=('u_prior_cvr', 'mean'),
    평균_remove_rate=('u_prior_remove_rate', 'mean'),
).reset_index()

x = np.arange(len(labels_t)); w4 = 0.26
axes[0].bar(x-w4, tenure_agg['평균_carts'],   w4, label='평균 담기', color='#42A5F5', alpha=0.85)
axes[0].bar(x,    tenure_agg['평균_views'],   w4, label='평균 조회', color='#66BB6A', alpha=0.85)
axes[0].bar(x+w4, tenure_agg['평균_removes'], w4, label='평균 삭제', color='#EF5350', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_t, fontproperties=fp, fontsize=9)
axes[0].set_title('활동 기간별 평균 이벤트 수', fontproperties=fp, fontsize=12)
axes[0].set_ylabel('평균 이벤트 수', fontproperties=fp)
axes[0].legend(prop=fp, fontsize=9)

ax2 = axes[1].twinx()
axes[1].plot(labels_t, tenure_agg['평균_prior_cvr']*100, 'o-', color=BC, lw=2.5, ms=9, label='u_prior_cvr (%)')
ax2.plot(labels_t, tenure_agg['평균_remove_rate']*100, 's--', color=RC, lw=2.5, ms=9, label='u_prior_remove_rate (%)')
axes[1].set_title('활동 기간별 전환율 / 삭제율 추이', fontproperties=fp, fontsize=12)
axes[1].set_ylabel('u_prior_cvr (%)', fontproperties=fp, color=BC)
ax2.set_ylabel('u_prior_remove_rate (%)', fontproperties=fp, color=RC)
lines1, lab1 = axes[1].get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1+lines2, lab1+lab2, prop=fp, fontsize=9)

bars_c = axes[2].bar(labels_t, tenure_agg['유저수'], color='#AB47BC', alpha=0.75, edgecolor='white')
axes[2].set_ylabel('유저 수', fontproperties=fp, color='#6A1B9A')
ax3 = axes[2].twinx()
ax3.plot(labels_t, tenure_agg['평균_상품수'], 'D-', color='#FF7043', lw=2.5, ms=9)
ax3.set_ylabel('평균 담은 상품 수', fontproperties=fp, color='#FF7043')
axes[2].set_title('활동 기간별 유저 수 및 평균 담은 상품 수', fontproperties=fp, fontsize=12)
for bar, cnt in zip(bars_c, tenure_agg['유저수']):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1000,
                 f'{cnt:,}', ha='center', fontproperties=fp, fontsize=8.5)

plt.tight_layout()
plt.savefig('fig3_tenure_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig3_tenure_breakdown.png 저장 완료")

# ── Fig 4: 진성 잠재 고객 프로파일 ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('"진성 잠재 고객" (7일+ & 4개+ 상품) 상세 프로파일',
             fontproperties=fp, fontsize=14, fontweight='bold')

act_tenure = active_serious['tenure_bin'].value_counts(normalize=True).sort_index() * 100
colors_at  = ['#BBDEFB','#90CAF9','#64B5F6','#42A5F5','#1E88E5']
bars_at = axes[0].bar(act_tenure.index, act_tenure.values, color=colors_at, edgecolor='white', linewidth=1.5)
axes[0].set_title(f'진성 잠재 고객 (n={len(active_serious):,}명)\n활동 기간 분포', fontproperties=fp, fontsize=12)
axes[0].set_ylabel('비율 (%)', fontproperties=fp)
for bar, val in zip(bars_at, act_tenure.values):
    n = int(len(active_serious) * val / 100)
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}%\n(n={n:,})', ha='center', fontproperties=fp, fontsize=9)

axes[1].axis('off')
cols = ['지표', '담고 삼', '담았지만\n안 삼', '진성 잠재\n고객']
rows = [
    ['유저 수',              f'{len(prof_bought):,}명',                f'{len(prof_cnb):,}명',                f'{len(active_serious):,}명'],
    ['평균 활동 기간',       f'{prof_bought["u_tenure_d"].mean():.1f}일', f'{prof_cnb["u_tenure_d"].mean():.1f}일', f'{active_serious["u_tenure_d"].mean():.1f}일'],
    ['평균 담은 상품 수',    f'{prof_bought["n_products"].mean():.1f}개', f'{prof_cnb["n_products"].mean():.1f}개', f'{active_serious["n_products"].mean():.1f}개'],
    ['평균 누적 담기',       f'{prof_bought["u_carts"].mean():.1f}회',   f'{prof_cnb["u_carts"].mean():.1f}회',   f'{active_serious["u_carts"].mean():.1f}회'],
    ['평균 누적 조회',       f'{prof_bought["u_views"].mean():.1f}회',   f'{prof_cnb["u_views"].mean():.1f}회',   f'{active_serious["u_views"].mean():.1f}회'],
    ['평균 세션 깊이',       f'{prof_bought["avg_s_depth"].mean():.1f}', f'{prof_cnb["avg_s_depth"].mean():.1f}', f'{active_serious["avg_s_depth"].mean():.1f}'],
    ['u_prior_cvr',         f'{prof_bought["u_prior_cvr"].mean():.3f}', f'{prof_cnb["u_prior_cvr"].mean():.3f}', f'{active_serious["u_prior_cvr"].mean():.3f}'],
    ['u_prior_remove_rate', f'{prof_bought["u_prior_remove_rate"].mean():.3f}', f'{prof_cnb["u_prior_remove_rate"].mean():.3f}', f'{active_serious["u_prior_remove_rate"].mean():.3f}'],
    ['never_purchased=True',f'{prof_bought["u_never_purchased"].mean()*100:.1f}%', f'{prof_cnb["u_never_purchased"].mean()*100:.1f}%', f'{active_serious["u_never_purchased"].mean()*100:.1f}%'],
    ['평균 가격',            f'${prof_bought["avg_price"].mean():.2f}', f'${prof_cnb["avg_price"].mean():.2f}', f'${active_serious["avg_price"].mean():.2f}'],
]
table = axes[1].table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)
for j in range(len(cols)):
    table[0,j].set_facecolor('#1565C0')
    table[0,j].set_text_props(color='white', fontproperties=fp, fontsize=10)
for i in range(1, len(rows)+1):
    for j, color in enumerate([None, '#E3F2FD', '#FFEBEE', '#E8F5E9']):
        if color:
            table[i,j].set_facecolor(color)
        table[i,j].get_text().set_fontproperties(fp)
axes[1].set_title('3개 그룹 핵심 지표 요약', fontproperties=fp, fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('fig4_serious_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("  fig4_serious_profile.png 저장 완료")

# ── Fig 5: 활동 기간 분포 비교 ────────────────────────────
labels_t2 = ['1일 이하', '1~7일', '7~30일', '30~60일', '60일 초과']
prof_bought['tenure_bin2'] = pd.cut(prof_bought['u_tenure_d'], bins=bins_t, labels=labels_t2)
prof_cnb['tenure_bin2']    = pd.cut(prof_cnb['u_tenure_d'],    bins=bins_t, labels=labels_t2)
bought_cnt = prof_bought['tenure_bin2'].value_counts().sort_index()
cnb_cnt    = prof_cnb['tenure_bin2'].value_counts().sort_index()
bought_pct = (bought_cnt / bought_cnt.sum() * 100).round(1)
cnb_pct    = (cnb_cnt    / cnb_cnt.sum()    * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('활동 기간(u_tenure_d) 분포 비교\n담고 삼 vs 담았지만 안 삼',
             fontproperties=fp, fontsize=15, fontweight='bold')
x = np.arange(len(labels_t2)); w = 0.38
for ax, vals_b, vals_c, ylabel, title in [
    (axes[0], bought_pct.values, cnb_pct.values, '비율 (%)', '비율 비교 (%)\n(각 그룹 내 비율)'),
    (axes[1], bought_cnt.values, cnb_cnt.values, '유저 수 (명)', '유저 수 비교 (명)'),
]:
    bars_b = ax.bar(x-w/2, vals_b, w, label=f'담고 삼 (n={len(prof_bought):,}명)', color=BC, alpha=0.88, edgecolor='white')
    bars_c = ax.bar(x+w/2, vals_c, w, label=f'담았지만 안 삼 (n={len(prof_cnb):,}명)', color=RC, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars_b, vals_b):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals_b)*0.01,
                f'{val:,.1f}%' if ylabel == '비율 (%)' else f'{val:,}', ha='center', fontproperties=fp, fontsize=9)
    for bar, val in zip(bars_c, vals_c):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals_c)*0.01,
                f'{val:,.1f}%' if ylabel == '비율 (%)' else f'{val:,}', ha='center', fontproperties=fp, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels_t2, fontproperties=fp, fontsize=10.5)
    ax.set_ylabel(ylabel, fontproperties=fp, fontsize=12)
    ax.set_title(title, fontproperties=fp, fontsize=12)
    ax.legend(prop=fp, fontsize=10)
    ax.set_ylim(0, max(max(vals_b), max(vals_c)) * 1.18)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('fig5_tenure_dist.png', dpi=180, bbox_inches='tight')
plt.close()
print("  fig5_tenure_dist.png 저장 완료")

# ── Fig 6: 담은 고유 상품 수 분포 비교 ────────────────────
labels_p2 = ['1개', '2~3개', '4~10개', '11~30개', '31~100개', '100개 초과']
bins_p2   = [0, 1, 3, 10, 30, 100, 2200]
prof_bought['prod_bin2'] = pd.cut(prof_bought['n_products'], bins=bins_p2, labels=labels_p2)
prof_cnb['prod_bin2']    = pd.cut(prof_cnb['n_products'],    bins=bins_p2, labels=labels_p2)
b_cnt2 = prof_bought['prod_bin2'].value_counts().sort_index()
c_cnt2 = prof_cnb['prod_bin2'].value_counts().sort_index()
b_pct2 = (b_cnt2 / b_cnt2.sum() * 100).round(1)
c_pct2 = (c_cnt2 / c_cnt2.sum() * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('담은 고유 상품 수 분포 비교\n담고 삼 vs 담았지만 안 삼',
             fontproperties=fp, fontsize=15, fontweight='bold')
x = np.arange(len(labels_p2)); w = 0.38
for ax, vals_b, vals_c, ylabel, title in [
    (axes[0], b_pct2.values, c_pct2.values, '비율 (%)', '비율 비교 (%)\n(각 그룹 내 비율)'),
    (axes[1], b_cnt2.values, c_cnt2.values, '유저 수 (명)', '유저 수 비교 (명)'),
]:
    bars_b = ax.bar(x-w/2, vals_b, w, label=f'담고 삼 (n={len(prof_bought):,}명)', color=BC, alpha=0.88, edgecolor='white')
    bars_c = ax.bar(x+w/2, vals_c, w, label=f'담았지만 안 삼 (n={len(prof_cnb):,}명)', color=RC, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars_b, vals_b):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals_b)*0.01,
                f'{val:,.1f}%' if ylabel == '비율 (%)' else f'{val:,}', ha='center', fontproperties=fp, fontsize=9)
    for bar, val in zip(bars_c, vals_c):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals_c)*0.01,
                f'{val:,.1f}%' if ylabel == '비율 (%)' else f'{val:,}', ha='center', fontproperties=fp, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels_p2, fontproperties=fp, fontsize=10.5)
    ax.set_ylabel(ylabel, fontproperties=fp, fontsize=12)
    ax.set_title(title, fontproperties=fp, fontsize=12)
    ax.legend(prop=fp, fontsize=10)
    ax.set_ylim(0, max(max(vals_b), max(vals_c)) * 1.18)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('fig6_products_dist.png', dpi=180, bbox_inches='tight')
plt.close()
print("  fig6_products_dist.png 저장 완료")

print("\n[Step 4 완료] 6개 차트 생성 완료")
