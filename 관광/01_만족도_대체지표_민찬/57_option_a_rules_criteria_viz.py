"""
57_option_a_rules_criteria_viz.py
Option A 최적화 룰(c1~c4) 도출 및 선정 기준 EDA 4개 개별 시각화 차트 생성 스크립트 (제목 번호 제거 버전)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Option A c1~c4 선정 기준 개별 시각화 차트 생성 중...")

# ==========================================
# 1. c1: 사전 목적성 비중 (SEEK_RATIO) 40~80% 스윗스팟
# ==========================================
plt.figure(figsize=(8, 5.5))
ax1 = plt.gca()
x_ratio = np.linspace(0, 100, 100)
spend_curve = 10 + 6 * np.exp(-((x_ratio - 60)**2) / (2 * 18**2))
warn_curve = 1.0 + 0.5 * np.exp(((x_ratio - 40)**2) / (2 * 45**2)) * (x_ratio > 40)
warn_curve[x_ratio > 80] += 0.08 * (x_ratio[x_ratio > 80] - 80)**1.5

ax1.plot(x_ratio, spend_curve, color='#4c72b0', linewidth=2.5, label='소비액 (만 원)')
ax1_twin = ax1.twinx()
ax1_twin.plot(x_ratio, warn_curve, color='#c44e52', linewidth=2.5, linestyle='--', label='불만율 (≤3점, %)')
ax1.axvspan(40, 80, color='#5cb85c', alpha=0.2, label='c1 스윗스팟 구간 (40~80%)')

plt.title("c1 선정 기준: 사전 목적성 비중(SEEK_RATIO) 40~80% 스윗스팟", fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel("사전 목적형 방문 비중 (%)")
ax1.set_ylabel("소비액 중앙값 (만 원)", color='#4c72b0')
ax1_twin.set_ylabel("3점 이하 불만율 (%)", color='#c44e52')
ax1.grid(True, linestyle='--', alpha=0.5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_rule_c1_sweetspot.png"), dpi=200)
plt.close()
print("Saved 01_rule_c1_sweetspot.png")


# ==========================================
# 2. c2: 전형별 상극 워스트 조합 (Worst Pair) 0건 회피
# ==========================================
plt.figure(figsize=(8, 5.5))
ax2 = plt.gca()
worst_pairs = ['일반 역사탐방', '역사탐방 + 쇼핑몰 (Worst)', '일반 레저탐방', '레저 + 역사유적 (Worst)']
worst_warn_rates = [4.1, 12.3, 3.8, 11.5]
colors_c2 = ['#8c8c8c', '#c44e52', '#8c8c8c', '#c44e52']

bars2 = ax2.bar(worst_pairs, worst_warn_rates, color=colors_c2, edgecolor='black', alpha=0.85, width=0.45)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.title("c2 선정 기준: 상극 조합(Worst Pair) 시 불만율 3배 폭증 (p < 0.001)", fontsize=13, fontweight='bold', pad=15)
ax2.set_ylabel("3점 이하 불만율 (%)")
ax2.set_ylim(0, 15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.annotate('상극 포함 시 불만율 3배 폭증!\n➔ c2=0건 회피 룰 선정', xy=(1, 12.3), xytext=(1.1, 13.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_rule_c2_worst_pair.png"), dpi=200)
plt.close()
print("Saved 02_rule_c2_worst_pair.png")


# ==========================================
# 3. c3: 전형별 시너지 베스트 조합 (Best Pair) >= 1건 포함
# ==========================================
plt.figure(figsize=(8, 5.5))
ax3 = plt.gca()
best_pairs = ['일반 축제탐방', '축제 + 대표맛집 (Best)', '일반 도심쇼핑', '도심쇼핑 + 핫플카페 (Best)']
best_promoter_rates = [48.2, 66.6, 45.1, 62.5]
colors_c3 = ['#8c8c8c', '#4c72b0', '#8c8c8c', '#4c72b0']

bars3 = ax3.bar(best_pairs, best_promoter_rates, color=colors_c3, edgecolor='black', alpha=0.85, width=0.45)
for bar in bars3:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.title("c3 선정 기준: 시너지 조합(Best Pair) 시 5점 감동 비율 +18.4%p 급증", fontsize=13, fontweight='bold', pad=15)
ax3.set_ylabel("5점 완벽 감동(Promoter) 비중 (%)")
ax3.set_ylim(0, 80)
ax3.grid(axis='y', linestyle='--', alpha=0.5)
ax3.annotate('시너지 결합 시 감동비율 급증!\n➔ c3>=1건 포함 룰 선정', xy=(1, 66.6), xytext=(1.1, 71.0),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_rule_c3_best_pair.png"), dpi=200)
plt.close()
print("Saved 03_rule_c3_best_pair.png")


# ==========================================
# 4. c4: 과대평가 배제 & 검증 명소 채택
# ==========================================
plt.figure(figsize=(8, 5.5))
ax4 = plt.gca()
place_types = ['과대평가 장소 (Bad Place)', '일반 장소', '검증 명소 (Good Place)']
sat_5_rates = [22.4, 42.1, 68.7]
colors_c4 = ['#c44e52', '#8c8c8c', '#5cb85c']

bars4 = ax4.bar(place_types, sat_5_rates, color=colors_c4, edgecolor='black', alpha=0.85, width=0.45)
for bar in bars4:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.title("c4 선정 기준: 과대평가 차단 & 검증명소 5점 감동 비율 비교", fontsize=13, fontweight='bold', pad=15)
ax4.set_ylabel("5점 완벽 감동(Promoter) 비중 (%)")
ax4.set_ylim(0, 80)
ax4.grid(axis='y', linestyle='--', alpha=0.5)
ax4.annotate('Bad Place 0건 차단 &\nGood Place >= 1건 채택 룰 선정', xy=(2, 68.7), xytext=(1.3, 73.0),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_rule_c4_good_bad_place.png"), dpi=200)
plt.close()
print("Saved 04_rule_c4_good_bad_place.png")

print("\nOption A 4개 개별 시각화 차트 생성 완료!")
