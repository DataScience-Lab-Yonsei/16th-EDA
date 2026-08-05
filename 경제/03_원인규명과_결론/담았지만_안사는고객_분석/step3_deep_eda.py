"""
Step 3. 심층 EDA 분석
======================
"담았지만 안 삼" 그룹의 비구매 원인을 다각도로 분석합니다.

분석 항목:
  1. 활동 기간 × 담은 상품 수 교차 세분화 → 진성 잠재 고객 식별
  2. 활동 기간별 상세 프로파일
  3. 삭제 행동 패턴 세분화
  4. 마지막 이벤트 시점 분석
  5. viewed_first 비율 분석
  6. 담은 상품의 p_prior_cvr 분포 비교
  7. 가격대별 분포 비교
  8. 월별 / 시간대별 비교
  9. 반복 행동 피쳐 비교
  10. 진성 잠재 고객 vs 담고 삼 상세 비교

사전 조건:
  - step2_classify_users.py 실행 후 prof_bought.csv, prof_cnb.csv 생성 필요

실행 방법:
    python step3_deep_eda.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = '../cosmetics_train.csv'

print("=" * 60)
print("Step 3. 심층 EDA 분석")
print("=" * 60)

print("\n데이터 로딩 중...")
df          = pd.read_csv(CSV_PATH)
df['t_dt']  = pd.to_datetime(df['t'])
prof_bought = pd.read_csv('prof_bought.csv')
prof_cnb    = pd.read_csv('prof_cnb.csv')

df_bought = df[df['user_hash'].isin(prof_bought['user_hash'])].copy()
df_cnb    = df[df['user_hash'].isin(prof_cnb['user_hash'])].copy()

OBS_END = pd.Timestamp('2020-02-22', tz='UTC')

# 구간 정의
bins_t   = [-0.001, 1, 7, 30, 60, 145]
labels_t = ['1일 이하', '1~7일', '7~30일', '30~60일', '60일 초과']
bins_p   = [0, 1, 3, 10, 50, 2200]
labels_p = ['1개', '2~3개', '4~10개', '11~50개', '51개+']

prof_cnb['tenure_bin'] = pd.cut(prof_cnb['u_tenure_d'], bins=bins_t, labels=labels_t)
prof_cnb['prod_bin']   = pd.cut(prof_cnb['n_products'],  bins=bins_p, labels=labels_p)

# ── [1] 활동 기간 × 상품 수 교차 세분화 ───────────────────
print("\n[1] 활동 기간 × 담은 상품 수 교차 세분화 (담았지만 안 삼)")
cross = prof_cnb.groupby(['tenure_bin', 'prod_bin'], observed=True).size().unstack(fill_value=0)
cross_pct = (cross.div(cross.sum().sum()) * 100).round(1)
print(cross_pct.to_string())

active_serious = prof_cnb[(prof_cnb['u_tenure_d'] >= 7) & (prof_cnb['n_products'] >= 4)]
print(f"\n진성 잠재 고객 (7일+ & 4개+ 상품): {len(active_serious):,}명 ({len(active_serious)/len(prof_cnb)*100:.1f}%)")
print(f"  평균 활동 기간:  {active_serious['u_tenure_d'].mean():.1f}일")
print(f"  평균 담은 상품:  {active_serious['n_products'].mean():.1f}개")
print(f"  평균 세션 깊이:  {active_serious['avg_s_depth'].mean():.1f}")
print(f"  u_never_purchased=True: {active_serious['u_never_purchased'].mean()*100:.1f}%")

# ── [2] 활동 기간별 상세 프로파일 ──────────────────────────
print("\n[2] 활동 기간별 상세 프로파일 (담았지만 안 삼)")
tenure_profile = prof_cnb.groupby('tenure_bin', observed=True).agg(
    유저수=('user_hash', 'count'),
    평균_carts=('u_carts', 'mean'),
    평균_views=('u_views', 'mean'),
    평균_removes=('u_removes', 'mean'),
    평균_상품수=('n_products', 'mean'),
    평균_세션깊이=('avg_s_depth', 'mean'),
    평균_prior_cvr=('u_prior_cvr', 'mean'),
    평균_remove_rate=('u_prior_remove_rate', 'mean'),
).round(2)
tenure_profile['비중(%)'] = (tenure_profile['유저수'] / tenure_profile['유저수'].sum() * 100).round(1)
print(tenure_profile.to_string())

# ── [3] 삭제 행동 패턴 세분화 ──────────────────────────────
print("\n[3] 삭제 행동 패턴 세분화 (담았지만 안 삼)")
prof_cnb['remove_bin'] = pd.cut(
    prof_cnb['remove_rate'],
    bins=[-0.001, 0, 0.25, 0.5, 0.75, 1.001],
    labels=['0%(방치만)', '0~25%', '25~50%', '50~75%', '75~100%(삭제위주)']
)
remove_profile = prof_cnb.groupby('remove_bin', observed=True).agg(
    유저수=('user_hash', 'count'),
    평균_tenure=('u_tenure_d', 'mean'),
    평균_carts=('u_carts', 'mean'),
    평균_상품수=('n_products', 'mean'),
    평균_prior_cvr=('u_prior_cvr', 'mean'),
    평균_가격=('avg_price', 'mean'),
).round(2)
remove_profile['비중(%)'] = (remove_profile['유저수'] / remove_profile['유저수'].sum() * 100).round(1)
print(remove_profile.to_string())

# ── [4] 마지막 이벤트 시점 분석 ───────────────────────────
print("\n[4] 마지막 이벤트 시점 분석 (u_recency_d 기준)")
bins_d   = [0, 7, 30, 60, 90, 145]
labels_d = ['7일 이내', '7~30일', '30~60일', '60~90일', '90일 초과']
cnb_rec   = pd.cut(prof_cnb['u_recency_d'].fillna(999),    bins=bins_d, labels=labels_d).value_counts(normalize=True).sort_index() * 100
bought_rec = pd.cut(prof_bought['u_recency_d'].fillna(999), bins=bins_d, labels=labels_d).value_counts(normalize=True).sort_index() * 100
timing_df = pd.DataFrame({'담고 삼(%)': bought_rec, '담았지만 안 삼(%)': cnb_rec}).round(1)
print(timing_df.to_string())

# ── [5] viewed_first 비율 ──────────────────────────────────
print("\n[5] viewed_first 비율 비교")
print(f"  담고 삼:        {df_bought['viewed_first'].mean()*100:.1f}%")
print(f"  담았지만 안 삼: {df_cnb['viewed_first'].mean()*100:.1f}%")

# ── [6] p_prior_cvr 분포 비교 ──────────────────────────────
print("\n[6] 담은 상품의 p_prior_cvr 분포 비교")
bins_pcvr   = [-0.001, 0.1, 0.15, 0.2, 0.25, 0.3, 1.001]
labels_pcvr = ['0~10%', '10~15%', '15~20%', '20~25%', '25~30%', '30%+']
cnb_pcvr    = pd.cut(df_cnb['p_prior_cvr'],    bins=bins_pcvr, labels=labels_pcvr).value_counts(normalize=True).sort_index() * 100
bought_pcvr = pd.cut(df_bought['p_prior_cvr'],  bins=bins_pcvr, labels=labels_pcvr).value_counts(normalize=True).sort_index() * 100
pcvr_df = pd.DataFrame({'담고 삼(%)': bought_pcvr, '담았지만 안 삼(%)': cnb_pcvr}).round(1)
print(pcvr_df.to_string())

# ── [7] 가격대별 분포 비교 ─────────────────────────────────
print("\n[7] 가격대별 분포 비교")
bins_pr   = [0, 1, 3, 5, 10, 20, 330]
labels_pr = ['$0~1', '$1~3', '$3~5', '$5~10', '$10~20', '$20+']
cnb_pr    = pd.cut(df_cnb['price'].dropna(),    bins=bins_pr, labels=labels_pr).value_counts(normalize=True).sort_index() * 100
bought_pr = pd.cut(df_bought['price'].dropna(),  bins=bins_pr, labels=labels_pr).value_counts(normalize=True).sort_index() * 100
pr_df = pd.DataFrame({'담고 삼(%)': bought_pr, '담았지만 안 삼(%)': cnb_pr}).round(1)
print(pr_df.to_string())

# ── [8] 월별 / 시간대별 비교 ──────────────────────────────
print("\n[8] 월별 비교")
if 'month' in df.columns:
    cnb_m    = df_cnb['month'].value_counts(normalize=True).sort_index() * 100
    bought_m = df_bought['month'].value_counts(normalize=True).sort_index() * 100
    month_df = pd.DataFrame({'담고 삼(%)': bought_m, '담았지만 안 삼(%)': cnb_m}).round(1)
    print(month_df.to_string())

# ── [9] 반복 행동 피쳐 비교 ────────────────────────────────
print("\n[9] 반복 행동 피쳐 비교")
print(f"  담고 삼:        up_carted_before=True {df_bought['up_carted_before'].mean()*100:.1f}%  up_bought_before=True {df_bought['up_bought_before'].mean()*100:.1f}%")
print(f"  담았지만 안 삼: up_carted_before=True {df_cnb['up_carted_before'].mean()*100:.1f}%  up_bought_before=True {df_cnb['up_bought_before'].mean()*100:.1f}%")

# ── [10] 진성 잠재 고객 vs 담고 삼 상세 비교 ──────────────
print("\n[10] 진성 잠재 고객 vs 담고 삼 상세 비교")
print(f"  진성 잠재 고객 수: {len(active_serious):,}명")
compare_cols2 = ['u_carts', 'u_views', 'u_removes', 'u_tenure_d', 'n_products',
                 'avg_s_depth', 'avg_s_elapsed', 'u_prior_cvr', 'u_prior_remove_rate',
                 'avg_price', 'viewed_first_rate', 'p_prior_cvr_mean']
print(f"\n{'피쳐':<22} {'담고 삼 (평균)':>16} {'진성 잠재 고객 (평균)':>22} {'비율':>8}")
print("-" * 72)
for col in compare_cols2:
    if col in prof_bought.columns and col in active_serious.columns:
        b = prof_bought[col].mean()
        a = active_serious[col].mean()
        ratio = a / b if b != 0 else float('nan')
        print(f"{col:<22} {b:>16.3f} {a:>22.3f} {ratio:>8.3f}")

print("\n[Step 3 완료]")
