"""
Step 1. 기초 EDA (Basic Exploratory Data Analysis)
=====================================================
cosmetics_train.csv의 기본 구조, 컬럼 타입, 결측치,
이벤트/유저/세션 규모, outcome 분포 등을 확인합니다.

실행 방법:
    python step1_basic_eda.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────
CSV_PATH = '../cosmetics_train.csv'   # 필요 시 수정

# ── 데이터 로드 ────────────────────────────────────────────
print("=" * 60)
print("Step 1. 기초 EDA")
print("=" * 60)
print("\n[1] 데이터 로딩 중...")
df = pd.read_csv(CSV_PATH)
print(f"  총 행 수: {len(df):,}")
print(f"  총 컬럼 수: {len(df.columns)}")

# ── 컬럼 정보 ──────────────────────────────────────────────
print("\n[2] 컬럼 정보")
print(df.dtypes.to_string())

# ── 결측치 ────────────────────────────────────────────────
print("\n[3] 결측치 현황")
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)
null_df = pd.DataFrame({'결측 수': null_counts, '결측률(%)': null_pct})
print(null_df[null_df['결측 수'] > 0].to_string())

# ── 기본 규모 ──────────────────────────────────────────────
print("\n[4] 데이터 규모")
print(f"  고유 유저 수:  {df['user_hash'].nunique():,}명")
print(f"  고유 상품 수:  {df['product_id'].nunique():,}개")
print(f"  고유 세션 수:  {df['user_session'].nunique():,}개")
if 'brand' in df.columns:
    print(f"  고유 브랜드 수: {df['brand'].nunique():,}개")

# ── 관측 기간 ──────────────────────────────────────────────
print("\n[5] 관측 기간")
df['t_dt'] = pd.to_datetime(df['t'])
print(f"  시작: {df['t_dt'].min()}")
print(f"  종료: {df['t_dt'].max()}")

# ── outcome 분포 ───────────────────────────────────────────
print("\n[6] outcome 분포 (행 단위)")
oc = df['outcome'].value_counts()
oc_pct = (oc / len(df) * 100).round(1)
oc_df = pd.DataFrame({'건수': oc, '비율(%)': oc_pct})
print(oc_df.to_string())

# ── 월별 이벤트 분포 ───────────────────────────────────────
print("\n[7] 월별 이벤트 분포")
if 'month' in df.columns:
    month_dist = df['month'].value_counts().sort_index()
    print(month_dist.to_string())

# ── 수치형 피쳐 기초 통계 ──────────────────────────────────
print("\n[8] 수치형 피쳐 기초 통계 (주요 컬럼)")
num_cols = ['price', 'u_carts', 'u_views', 'u_purchases', 'u_tenure_d',
            'u_prior_cvr', 'u_prior_remove_rate', 's_depth', 's_elapsed_s',
            'p_prior_cvr', 'cart_repeat_cnt']
num_cols = [c for c in num_cols if c in df.columns]
print(df[num_cols].describe().round(3).to_string())

print("\n[Step 1 완료]")
