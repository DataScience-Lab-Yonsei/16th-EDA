# 성범죄 회귀 결과 요약

입력: `New_anal_sex.csv` · n(변호사 비결측)=79

## 변호사별 outcome 비율

```
         count      mean
lawyer                  
public      15  0.600000
lawyer      26  0.500000
lawfirm     38  0.578947
```

## 모형 적합

```
            model  n  pseudo_r2    llr_p
        A_minimal 79     0.0743 0.327646
       B_extended 79     0.1021 0.351433
      C_with_year 64     0.1043 0.405662
    E_ref_private 79     0.0743 0.327646
D_agree_on_lawyer 79     0.1909 0.004592
 F_suspended_only 79     0.0117 0.989454
```

## 해석 주의
- 소표본 · 양형구간 단순화 · 키워드 태깅 · 선택편향
- 변호사 효과가 비유의여도 ‘효과 없음’ 단정은 금지 (검정력 부족 가능)

상세 OR 표: `output/logit_*.csv`
