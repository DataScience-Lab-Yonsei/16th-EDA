# Dataset

## 원본 데이터 (출처)
| 데이터 | 설명 | 출처 |
|---|---|---|
| 기상청 종관기상관측 **ASOS** 일자료 | 전국 종관관측소 기온·습도·풍속·강수 등 | https://data.kma.go.kr |
| 기상청 방재기상관측 **AWS** 자료 | 방재기상관측망 관측 자료 | https://data.kma.go.kr |
| 한국환경공단 **에어코리아** 대기질 확정자료(2023–2025) | PM2.5·PM10·O₃·NO₂·SO₂·CO 농도 | https://www.airkorea.or.kr |
| 한국관광공사 **한국관광 데이터랩** 관광지표(2025) | 시군구별 방문자·관광소비·다양성·외지인 비중 | https://datalab.visitkorea.or.kr |

> 원본 관측 자료는 용량이 커(수십~수백 MB) 저장소에는 올리지 않고, 위 포털 링크로 대체합니다.
> 아래는 전처리·분석을 거친 **가공 데이터(분석용)** 샘플입니다.

## 포함된 가공 데이터
| 파일 | 설명 |
|---|---|
| `master_monthly_ktci_tourism.csv` | 시군구×월 날씨(정규화·KTCI)×관광지표 **분석용 마스터** |
| `sample_weather_master_ktci.csv` | 일별 마스터(KTCI·원관측값 포함) **샘플 5,000행** (전체는 용량 관계로 생략) |
| `hidden_by_season.csv` | 계절별 히든스팟 상위 후보 (매력−혼잡 점수) |
| `validation_summary.csv` | 검증(순열검정·부트스트랩·연도 안정성) 요약 |
| `spot_metrics.json` | 히든스팟 12곳의 계절 평균 지표(KTCI·기온·습도·풍속·강수·초미세) |

## 주요 컬럼 (master_monthly)
`signguCode·signguNm·baseYm` / `KTCI_data·KTCI_2014_adapted·KTCI_hybrid` /
`S_temp·S_humidity·S_wind·S_precip·S_airquality`(정규화 스트레스) /
`visitor_total` / `관광객다양성·관광소비강도·타권역방문자비중` 등
