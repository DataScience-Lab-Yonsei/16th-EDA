# 분석 단계별 입출력

## 01. 원자료 정제

- 입력: `data/raw/crime.zip`
- 주요 처리: 이유부 추출, HTML 제거, 날짜 보정, 심급·범죄군 분류
- 출력:
  - `data/interim/criminal_panel.parquet`
  - `results/tables/data_quality.csv`
  - `results/tables/year_counts.csv`
  - `results/tables/decade_composition.csv`
  - `results/tables/text_qc_flags.csv`

## 02. 판결문별 문체 지표

- 입력: `data/interim/criminal_panel.parquet`
- 주요 처리: Kiwi 형태소 분석, 문장 분리, 명사·동사·서술어·조사 비율,
  명사성, 문장당 어절·글자 수, 한자 비율, 표현 변이 계산
- 출력:
  - `data/processed/document_metrics.parquet`
  - `results/tables/metric_audit.csv`
  - `results/tables/metric_errors.csv`

## 03. 기본 시계열 분석

- 입력: 판결문별 지표
- 주요 처리: 연도별 원시 추세, 구성 표준화, 문서 단위 회귀,
  Benjamini–Hochberg 보정, 표현 변이와 탐색적 변화점
- 출력:
  - `annual_raw_metrics.csv`
  - `annual_standardized_metrics.csv`
  - `trend_models.csv`
  - `breakpoint_candidates.csv`
  - `variant_annual.csv`
  - `variant_summary.csv`

## 04–05. 기본 그림과 사례

- 기본 그림 F1–F6 생성
- 범죄군·심급·법원·문서 길이를 맞춘 과거·최근 사례 추출

## 06–08. 강건성 및 품질 점검

- 범죄군·심급 하위집단 추세
- 연도 고정효과 비교
- 공통 셀 균형 재표집
- 위약검정과 변화점 부트스트랩
- 긴 문장과 표현 변이의 원문 문맥 점검
- 강건성 그림 F7–F9 생성

## 09. 회귀선 및 추가 민감도

- 연도별 보정값의 선형 회귀선
- Newey–West HAC 표준오차와 95% 신뢰구간
- 문서 중앙값·IQR과 평균 비교
- 문서평균과 토큰·문장 풀링 집계 비교
- 문서 길이 통제 포함·제외 결과 비교
- 추가 그림 F10–F12 생성

## 재현 시 참고

- 모든 명령은 압축을 푼 최상위 폴더에서 실행하는 것을 권장합니다.
- `02_extract_metrics.py`는 체크포인트를 저장하므로 중단 후 재실행할 수 있습니다.
- 표와 그림은 기존 파일을 같은 이름으로 갱신할 수 있습니다.

