# hidden_spot — 계절별 히든스팟 도출

전처리·마스터 결합 결과(`../preprocess_code/` 산출)를 입력으로, 계절마다
**안 붐비고(혼잡도↓) · 날씨 좋고(KTCI ≥ 계절 중앙값) · 볼거리 있는(매력도↑)** 시군구를 뽑습니다. (백가은)

## 점수 정의

```
혼잡도  crowd  = z( log(방문객수) )                                  # 낮을수록 여유
매력도  appeal = z(관광객다양성) + z(관광소비강도) + z(타권역방문자비중)   # 세 지표 동일 가중
히든점수 hidden = appeal − crowd                                     # 좋은 날씨 통과 지역 중 상위
```

단위가 다른 지표를 표준화(z)로 같은 잣대에 맞춘 뒤, **계절마다 따로** 순위화합니다.
좋은 날씨(KTCI ≥ 계절 중앙값)와 비혼잡(방문객 ≤ 계절 중앙값) 필터를 통과한 후보 중 상위를 선정.

## 실행

```bash
pip install pandas numpy
python 03_hidden_spot.py --master ../Dataset/master_monthly_ktci_tourism.csv --ktci KTCI_hybrid
```

| 입력 | `../Dataset/master_monthly_ktci_tourism.csv` (날씨×관광 월별 마스터) |
|---|---|
| 출력 | `hidden_by_season.csv` (계절별 상위 후보) |

> KTCI 컬럼은 `KTCI_data / KTCI_2014_adapted / KTCI_hybrid` 중 선택 가능(`--ktci`).
