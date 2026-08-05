"""
Step 2. 고객 유형 분류 및 유저 프로파일 생성
=============================================
cart / purchase 이벤트 기준으로 고객을 분류하고,
유저 단위 집계 프로파일을 생성하여 CSV로 저장합니다.

분류 기준:
  - 담고 삼     : outcome에 'purchased'가 있는 유저 (cart O, purchase O)
  - 담았지만 안 삼: outcome에 'purchased'가 없는 유저 (cart O, purchase X)
  ※ CSV는 cart 이벤트 기준 전처리 데이터이므로
    '담지도 사지도 않음'과 '안 담고 구매'는 포함되지 않음.
    (Notion 원본 기준: 담지도 사지도 않음 247,566명 / 안 담고 구매 183명)

출력 파일:
  - prof_bought.csv : 담고 삼 유저 프로파일
  - prof_cnb.csv    : 담았지만 안 삼 유저 프로파일

실행 방법:
    python step2_classify_users.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = '../cosmetics_train.csv'

print("=" * 60)
print("Step 2. 고객 유형 분류 및 유저 프로파일 생성")
print("=" * 60)

print("\n데이터 로딩 중...")
df = pd.read_csv(CSV_PATH)

# ── 고객 유형 분류 ─────────────────────────────────────────
user_outcomes  = df.groupby('user_hash')['outcome'].apply(set)
has_cart       = set(df['user_hash'].unique())
has_purchase   = set(user_outcomes[user_outcomes.apply(lambda s: 'purchased' in s)].index)

type_bought = has_cart & has_purchase       # 담고 삼
type_cnb    = has_cart - has_purchase       # 담았지만 안 삼

print(f"\n[고객 유형 분류 결과]")
print(f"  담고 삼 (cart O, purchase O): {len(type_bought):,}명")
print(f"  담았지만 안 삼 (cart O, purchase X): {len(type_cnb):,}명")
print(f"  전체 유저: {len(has_cart):,}명")

df_bought = df[df['user_hash'].isin(type_bought)].copy()
df_cnb    = df[df['user_hash'].isin(type_cnb)].copy()

# ── 유저 단위 집계 함수 ────────────────────────────────────
def build_user_profile(df_sub):
    agg = df_sub.groupby('user_hash').agg(
        n_rows              = ('outcome',              'count'),
        n_purchased         = ('outcome',              lambda x: (x == 'purchased').sum()),
        n_removed           = ('outcome',              lambda x: (x == 'explicitly_removed').sum()),
        n_abandoned         = ('outcome',              lambda x: (x == 'silently_abandoned').sum()),
        u_carts             = ('u_carts',              'last'),
        u_views             = ('u_views',              'last'),
        u_purchases         = ('u_purchases',          'last'),
        u_removes           = ('u_removes',            'last'),
        u_spend             = ('u_spend',              'last'),
        u_tenure_d          = ('u_tenure_d',           'last'),
        u_recency_d         = ('u_recency_d',          'last'),
        u_prior_cvr         = ('u_prior_cvr',          'last'),
        u_prior_remove_rate = ('u_prior_remove_rate',  'last'),
        u_orders            = ('u_orders',             'last'),
        u_never_purchased   = ('u_never_purchased',    'last'),
        n_products          = ('product_id',           'nunique'),
        avg_price           = ('price',                'mean'),
        med_price           = ('price',                'median'),
        avg_s_depth         = ('s_depth',              'mean'),
        avg_s_elapsed       = ('s_elapsed_s',          'mean'),
        viewed_first_rate   = ('viewed_first',         'mean'),
        cart_repeat_cnt_mean= ('cart_repeat_cnt',      'mean'),
        p_prior_cvr_mean    = ('p_prior_cvr',          'mean'),
    ).reset_index()
    agg['remove_rate'] = agg['n_removed'] / agg['n_rows']
    return agg

print("\n유저 단위 집계 중...")
prof_bought = build_user_profile(df_bought)
prof_cnb    = build_user_profile(df_cnb)

prof_bought.to_csv('prof_bought.csv', index=False)
prof_cnb.to_csv('prof_cnb.csv',    index=False)
print("  prof_bought.csv 저장 완료")
print("  prof_cnb.csv 저장 완료")

# ── 기초 비교 출력 ─────────────────────────────────────────
print("\n[유저 단위 평균 비교]")
compare_cols = [
    ('n_rows',           '이벤트 수'),
    ('u_carts',          '누적 담기 수'),
    ('u_views',          '누적 조회 수'),
    ('u_removes',        '누적 삭제 수'),
    ('u_tenure_d',       '활동 기간(일)'),
    ('n_products',       '담은 고유 상품 수'),
    ('avg_s_depth',      '평균 세션 깊이'),
    ('avg_s_elapsed',    '평균 세션 경과(초)'),
    ('u_prior_cvr',      'u_prior_cvr'),
    ('u_prior_remove_rate', 'u_prior_remove_rate'),
    ('avg_price',        '평균 가격'),
    ('remove_rate',      '삭제 비율'),
]
print(f"\n{'피쳐':<22} {'담고 삼 (평균)':>16} {'담았지만 안 삼 (평균)':>22}")
print("-" * 64)
for col, label in compare_cols:
    b = prof_bought[col].mean()
    c = prof_cnb[col].mean()
    print(f"{label:<22} {b:>16.3f} {c:>22.3f}")

print("\n[outcome 분포 비교 (행 단위)]")
for grp, label in [(df_bought, '담고 삼'), (df_cnb, '담았지만 안 삼')]:
    oc = grp['outcome'].value_counts(normalize=True) * 100
    print(f"\n  [{label}]")
    for k in ['purchased', 'explicitly_removed', 'silently_abandoned']:
        print(f"    {k}: {oc.get(k, 0):.1f}%")

print("\n[Step 2 완료]")
