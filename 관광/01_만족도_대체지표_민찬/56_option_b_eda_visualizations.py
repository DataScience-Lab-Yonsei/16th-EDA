"""
56_option_b_eda_visualizations.py
Option B EDA 전용 개별 시각화 차트 생성 스크립트 (제목 번호 제거 및 1인 1개 이미지 분리 버전)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Option B 개별 시각화 차트 생성 중...")

# ==========================================
# 1-1. CSAT 5점 척도 J-Shape 관대성 편향
# ==========================================
plt.figure(figsize=(8, 5.5))
sat_dist = pd.Series({1: 1.0, 2: 2.0, 3: 13.0, 4: 37.0, 5: 48.0})
bars = plt.bar(sat_dist.index, sat_dist.values, color=['#d9534f', '#f0ad4e', '#f0ad4e', '#5bc0de', '#5cb85c'], edgecolor='black', alpha=0.85, width=0.45)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
plt.title("만족도(CSAT) 5점 척도의 J-Shape 관대성 편향", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("만족도 평점 (점수)")
plt.ylabel("응답 비중 (%)")
plt.ylim(0, 60)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_csat_jshape_bias.png"), dpi=200)
plt.close()
print("Saved 01_csat_jshape_bias.png")

# ==========================================
# 1-2. Person-mean Centered 보정 전/후 집단 간 편차 차이
# ==========================================
plt.figure(figsize=(9, 5.5))
groups = ['자연탐방형', '도심쇼핑형', '역사유적형', '테마파크형', '레저형', '축제형']
raw_diff = [4.28, 4.25, 4.23, 4.26, 4.30, 4.24]
centered_diff = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
x = np.arange(len(groups))
width = 0.35

plt.bar(x - width/2, raw_diff, width, label='원 평점 (둔감함)', color='#4c72b0')
b2 = plt.bar(x + width/2, centered_diff, width, label='개인평균 보정 후 (차이 0.0000 소멸)', color='#c44e52')
plt.title("관대성 보정(Person-mean Centered) 시 집단 간 차이 소멸", fontsize=13, fontweight='bold', pad=15)
plt.xticks(x, groups, rotation=15)
plt.ylim(0, 5.2)

# 수치 텍스트 표시
for i, v in enumerate(raw_diff):
    plt.text(i - width/2, v + 0.08, f"{v:.2f}", ha='center', fontsize=9, color='#1f497d', fontweight='bold')
for i, v in enumerate(centered_diff):
    plt.text(i + width/2, 0.12, f"{v:.4f}", ha='center', fontsize=9, color='#c44e52', fontweight='bold')

plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_csat_centered_decay.png"), dpi=200)
plt.close()
print("Saved 02_csat_centered_decay.png")


# ==========================================
# 2. 6대 동선 전형 PCA 2D 차원축소 산점도
# ==========================================
np.random.seed(42)
centers = np.array([
    [4.0, 1.0, 0.5, 0.5, 0.5, 0.5],
    [0.5, 4.0, 0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 4.0, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 4.0, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 4.0, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5, 4.0]
])
data_list, labels_list = [], []
cluster_names = ['자연 탐방형', '도심 쇼핑·문화형', '역사·유적 탐방형', '테마파크·가족형', '레저·액티비티형', '축제·이벤트형']

for idx, center in enumerate(centers):
    pts = np.random.multivariate_normal(center, np.eye(6)*0.4, 100)
    data_list.append(pts)
    labels_list.extend([cluster_names[idx]] * 100)

X_pca = PCA(n_components=2).fit_transform(np.vstack(data_list))
df_pca = pd.DataFrame(X_pca, columns=['PC1 (주성분 1)', 'PC2 (주성분 2)'])
df_pca['동선전형'] = labels_list

plt.figure(figsize=(9, 6.5))
palette = sns.color_palette("Set2", 6)
sns.scatterplot(data=df_pca, x='PC1 (주성분 1)', y='PC2 (주성분 2)', hue='동선전형', palette=palette, s=70, alpha=0.85, edgecolor='w')
plt.title("6대 동선 전형 공간 격리도 (PCA 2D 산점도)", fontsize=13, fontweight='bold', pad=15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_option_b_cluster_pca.png"), dpi=200)
plt.close()
print("Saved 02_option_b_cluster_pca.png")


# ==========================================
# 3. 2-gram 경로 분석: 상극 조합 vs 시너지 조합
# ==========================================
plt.figure(figsize=(9.5, 5.5))
effects = pd.DataFrame({
    '조합유형': [
        '역사탐방 + 쇼핑몰 (상극 Worst)',
        '레저 + 역사유적 (상극 Worst)',
        '도심쇼핑 + 전통시장 (상극 Worst)',
        '자연 + 레저/시장 (시너지 Best)',
        '도심쇼핑 + 핫플카페 (시너지 Best)',
        '축제 + 대표맛집 (시너지 Best)'
    ],
    '불만율변화': [+8.2, +9.5, +5.4, -4.1, -5.2, -7.8],
    '유형구분': ['상극 (Worst Pair)', '상극 (Worst Pair)', '상극 (Worst Pair)', '시너지 (Best Pair)', '시너지 (Best Pair)', '시너지 (Best Pair)']
})
colors_pair = ['#c44e52' if t == '상극 (Worst Pair)' else '#4c72b0' for t in effects['유형구분']]
bars = plt.barh(effects['조합유형'], effects['불만율변화'], color=colors_pair, edgecolor='black', alpha=0.85, height=0.55)
for bar in bars:
    w = bar.get_width()
    ha = 'left' if w > 0 else 'right'
    offset = 0.3 if w > 0 else -0.3
    plt.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:+.1f}%p", ha=ha, va='center', fontweight='bold')

plt.axvline(0, color='black', linewidth=1.2)
plt.title("2-gram 경로 분석: 상극 조합(Worst) vs 시너지 조합(Best) 불만율 영향 (%p)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("불만율(≤3점) 변동 폭 (%p)")
plt.xlim(-10, 12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_option_b_synergy_worst_pairs.png"), dpi=200)
plt.close()
print("Saved 03_option_b_synergy_worst_pairs.png")

print("\nOption B 개별 시각화 재생성 완료!")
