#!/usr/bin/env python
# coding: utf-8
# 제출본 주기: 데이터 경로만 ../Dataset/ 상대경로로 수정했으며 분석 로직은 원본 그대로다.

# In[63]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("../Dataset/Breach.csv")

df = df.dropna(subset=['outcome'])
if df['outcome'].dtype == 'bool' or df['outcome'].dtype == 'object':
    df['outcome'] = df['outcome'].astype(int)


model_vars = [
    'outcome',
    'lawyer',
    'month',
    'conceal',
    'reflection',
    'settlement',
    'restitution',
]

df_model = df[model_vars].dropna()


formula = 'outcome ~ C(lawyer) + month + conceal + reflection + settlement + restitution'

model = smf.logit(formula, data=df_model)
result = model.fit(maxiter=1000)

print('==================== [로지스틱 회귀분석 결과 요약] ====================')
print(result.summary())

summary_df = pd.DataFrame(
    {
        'Coefficient (계수)': result.params,
        'Odds Ratio (오즈비)': np.exp(result.params),  
        'P-Value': result.pvalues,
        'Significant (p < 0.05)': result.pvalues < 0.05,
    }
)

print('\n================ [계수, 오즈비 및 p-값 통합 테이블] ================')
print(summary_df)


# In[56]:


file_path = '../Dataset/Fraud.csv.csv'
df = pd.read_csv("../Dataset/Fraud.csv")

df.columns = df.columns.str.replace(r'.*outcome.*', 'outcome', regex=True)
df.columns = [c.strip().split('.')[-1] for c in df.columns]

numeric_cols = [
    'settlement',
    'restitution',
    'conceal',
    'reflection',
    'month',
    'demage',
    'repeated',
]
for col in numeric_cols:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if 'outcome' in df.columns:
  df = df.dropna(subset=['outcome'])
  df['outcome'] = df['outcome'].astype(int)



print('================== [재판 결과(outcome) 회귀분석 (기준: public)] ==================')
formula_outcome = (
    'outcome ~ C(lawyer, Treatment(reference=\'public\')) + month + demage +'
    ' conceal + reflection + settlement + restitution'
)

model_vars = [
    'outcome',
    'lawyer',
    'month',
    'demage',
    'conceal',
    'reflection',
    'settlement',
    'restitution',
]
df_model = df[model_vars].dropna()

model = smf.logit(formula_outcome, data=df_model)
result = model.fit(maxiter=1000)

print(result.summary())

summary_df = pd.DataFrame(
    {
        'Coefficient': result.params,
        'Odds Ratio': np.exp(result.params),
        'P-Value': result.pvalues,
        'Significant (p < 0.05)': result.pvalues < 0.05,
    }
)
print('\n[요약 테이블]')
print(summary_df)


# In[57]:


file_path = '../Dataset/Fraud.csv'  # 실제 파일 경로
df = pd.read_csv(file_path)


df.columns = df.columns.str.replace(r'.*outcome.*', 'outcome', regex=True)
df.columns = [c.strip().split('.')[-1] for c in df.columns]

for col in ['settlement', 'restitution', 'conceal', 'reflection', 'month', 'demage', 'repeated']:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if 'outcome' in df.columns:
  df = df.dropna(subset=['outcome'])
  df['outcome'] = df['outcome'].astype(int)


print('================== [모델 1: 재판 결과(outcome) 회귀분석] ==================')

formula_1 = (
    'outcome ~ C(lawyer) + month + demage + conceal + reflection + settlement'
    ' + restitution'
)
model_vars_1 = [
    'outcome',
    'lawyer',
    'month',
    'demage',
    'conceal',
    'reflection',
    'settlement',
    'restitution',
]
df_model_1 = df[model_vars_1].dropna()

model_1 = smf.logit(formula_1, data=df_model_1)
result_1 = model_1.fit(maxiter=1000)

print(result_1.summary())

summary_df_1 = pd.DataFrame(
    {
        'Coefficient': result_1.params,
        'Odds Ratio': np.exp(result_1.params),
        'P-Value': result_1.pvalues,
        'Significant (p < 0.05)': result_1.pvalues < 0.05,
    }
)
print('\n[모델 1 요약 테이블]')
print(summary_df_1)


print(
    '\n\n================== [모델 2: 합의여부(settlement) 회귀분석]'
    ' =================='
)

formula_2 = (
    'settlement ~ C(lawyer) + month + demage + conceal + reflection +'
    ' restitution'
)
model_vars_2 = [
    'settlement',
    'lawyer',
    'month',
    'demage',
    'conceal',
    'reflection',
    'restitution',
]
df_model_2 = df[model_vars_2].dropna()

model_2 = smf.logit(formula_2, data=df_model_2)
result_2 = model_2.fit(maxiter=1000)

print(result_2.summary())

summary_df_2 = pd.DataFrame(
    {
        'Coefficient': result_2.params,
        'Odds Ratio': np.exp(result_2.params),
        'P-Value': result_2.pvalues,
        'Significant (p < 0.05)': result_2.pvalues < 0.05,
    }
)
print('\n[모델 2 요약 테이블]')
print(summary_df_2)


# In[58]:


file_path = '../Dataset/Fraud.csv.csv' 
df = pd.read_csv("../Dataset/Fraud.csv")

df.columns = df.columns.str.replace(r".*outcome.*", "outcome", regex=True)
df.columns = [c.strip().split(".")[-1] for c in df.columns]

numeric_cols = ['month', 'demage', 'repeated', 'reflection', 'conceal', 'settlement', 'restitution']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'lawyer' in df.columns:
    df['is_lawfirm'] = (df['lawyer'] == 'lawfirm').astype(int)


print('================== [피해액 구간별 변호사 선임 현황] ==================')
if 'demage' in df.columns and 'lawyer' in df.columns:
    
    df['demage_quantile'] = pd.qcut(df['demage'], q=4, labels=['1사분위(소)', '2사분위', '3사분위', '4사분위(대)'])
    
   
    crosstab_result = pd.crosstab(df['demage_quantile'], df['lawyer'], normalize='index') * 100
    print(crosstab_result.round(2))


print('\n\n================== [로지스틱 회귀: 피해액이 로펌 선임에 미치는 영향] ==================')

formula_lawfirm = 'is_lawfirm ~ demage + month + reflection + conceal + settlement + restitution'
model_vars = ['is_lawfirm', 'demage', 'month', 'reflection', 'conceal', 'settlement', 'restitution']

df_model = df[model_vars].dropna()

model_lawfirm = smf.logit(formula_lawfirm, data=df_model)
result_lawfirm = model_lawfirm.fit(maxiter=1000)

print(result_lawfirm.summary())

summary_lawfirm = pd.DataFrame({
    'Coefficient': result_lawfirm.params,
    'Odds Ratio': np.exp(result_lawfirm.params),
    'P-Value': result_lawfirm.pvalues,
    'Significant (p < 0.05)': result_lawfirm.pvalues < 0.05
})

print('\n[로펌 선임 여부 모델 요약 테이블]')
print(summary_lawfirm)


# In[59]:


df = pd.read_csv("../Dataset/Fraud.csv")


df.columns = df.columns.str.replace(r'.*outcome.*', 'outcome', regex=True)
df.columns = [c.strip().split('.')[-1] for c in df.columns]

numeric_cols = [
    'settlement',
    'restitution',
    'conceal',
    'reflection',
    'month',
    'demage',
    'repeated',
]
for col in numeric_cols:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


print(
    '================== [합의여부(settlement) 회귀분석 (기준:'
    ' public)] =================='
)


formula_settlement = (
    "settlement ~ C(lawyer, Treatment(reference='public')) + month + demage"
    ' + conceal + reflection + restitution'
)

model_vars = [
    'settlement',
    'lawyer',
    'month',
    'demage',
    'conceal',
    'reflection',
    'restitution',
]
df_model = df[model_vars].dropna()


model_settlement = smf.logit(formula_settlement, data=df_model)
result_settlement = model_settlement.fit(maxiter=1000)

print(result_settlement.summary())


summary_df = pd.DataFrame(
    {
        'Coefficient': result_settlement.params,
        'Odds Ratio': np.exp(result_settlement.params),
        'P-Value': result_settlement.pvalues,
        'Significant (p < 0.05)': result_settlement.pvalues < 0.05,
    }
)

print('\n[합의여부 모델 요약 테이블]')
print(summary_df)


# In[60]:


df = pd.read_csv("../Dataset/Usedata.csv")

df.columns = df.columns.str.replace(r'.*outcome.*', 'outcome', regex=True)
df.columns = [c.strip().split('.')[-1] for c in df.columns]

numeric_cols = [
    'outcome',
    'month',
    'demage',
    'repeated',
    'reflection',
    'conceal',
    'settlement',
    'restitution',
]
for col in numeric_cols:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if 'outcome' in df.columns:
  df_clean = df.dropna(subset=['outcome']).copy()
  df_clean['outcome'] = df_clean['outcome'].astype(int)


print(
    '================== [최종 통합 데이터 회귀분석 (기준: public)]'
    ' =================='
)


formula_final = (
    "outcome ~ C(lawyer, Treatment(reference='public')) + C(crime) + month +"
    ' demage + conceal + reflection + settlement + restitution'
)

model_vars = [
    'outcome',
    'lawyer',
    'crime',
    'month',
    'demage',
    'conceal',
    'reflection',
    'settlement',
    'restitution',
]
df_model = df_clean[model_vars].dropna()


final_model = smf.logit(formula_final, data=df_model)
final_result = final_model.fit(maxiter=10000)

print(final_result.summary())


summary_final_df = pd.DataFrame(
    {
        'Coefficient': final_result.params,
        'Odds Ratio': np.exp(final_result.params),
        'P-Value': final_result.pvalues,
        'Significant (p < 0.05)': final_result.pvalues < 0.05,
    }
)

print('\n[최종 통합 모델 요약 테이블]')
print(summary_final_df)

