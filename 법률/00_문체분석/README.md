# 판결문 문체 통시 분석 코드 공유본

1980–2025년 공개 형사 판결문 이유부의 문체 변화를 분석한 재현용 코드입니다.

이 압축파일은 코드 검토와 재현에 필요한 항목만 담았습니다. 원자료, 중간 데이터,
분석 결과물, 가상환경 및 개인 PC 경로는 포함하지 않았습니다.

## 1. 가장 빠른 실행 방법

1. Python 3.12 환경을 준비합니다.
2. `crime.zip`을 `data/raw/crime.zip`에 둡니다.
3. PowerShell에서 다음을 실행합니다.

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\run_analysis.ps1
```

특정 Python 실행 파일을 사용하려면 다음처럼 지정할 수 있습니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_analysis.ps1 `
  -PythonExe "C:\경로\python.exe"
```

## 2. 코드 구성

| 순서 | 파일 | 역할 |
|---:|---|---|
| 공통 | `src/common.py` | 공통 경로, 다중검정 보정, 그림 설정 |
| 01 | `src/01_prepare_data.py` | ZIP 확인, 이유부 추출, 날짜·심급·범죄군 정리, 품질 점검 |
| 02 | `src/02_extract_metrics.py` | Kiwi 형태소 분석, 문장 길이와 품사·한자·표현 지표 계산 |
| 03 | `src/03_analyze_timeseries.py` | 연도별 원시·보정 추세, 회귀, 변화점, 표현 변이 분석 |
| 04 | `src/04_create_figures.py` | 표본 구성과 주요 추세 등 기본 그림 생성 |
| 05 | `src/05_extract_examples.py` | 동일 조건의 과거·최근 판결문 사례 추출 |
| 06 | `src/06_robustness.py` | 범죄군·심급 하위집단, 연도 고정효과, 균형 재표집, 위약검정 |
| 07 | `src/07_create_robustness_figures.py` | 강건성 분석 그림 생성 |
| 08 | `src/08_qc_context.py` | 1980년대 긴 문장과 표현 변이 문맥의 수작업 점검 보조 |
| 09 | `src/09_supplementary_analysis.py` | 회귀선·HAC, 중앙값·IQR, 집계·길이 통제 민감도 |

표현 변이쌍은 `config/variant_pairs.csv`에서 관리합니다.

## 3. 분석 흐름

```text
crime.zip
   ↓
01 정제·표본 구성
   ↓
02 판결문별 문체 지표
   ↓
03 기본 시계열·보정 회귀
   ├─ 04 기본 그림
   ├─ 05 동일 조건 사례
   ├─ 06 강건성·위약·변화점
   │    └─ 07 강건성 그림
   ├─ 08 문맥·품질 점검
   └─ 09 회귀선·집계 민감도
```

세부 입출력은 `docs/ANALYSIS_FLOW.md`를 참고하십시오.

## 4. 주요 산출 위치

- 판결문별 지표: `data/processed/document_metrics.parquet`
- 통계표: `results/tables/`
- 그림: `results/figures/`
- 문장 사례: `results/examples/`

## 5. 분석 기준

- 주 분석 기간: 1980–2025년
- 관측 단위: 판결문 1건
- 주요 통제: 심급, 범죄군, 법원, 문서 길이 및 구성 변화
- 표준오차: 법원 군집 강건 표준오차 또는 연도 시계열 HAC
- 다중검정: Benjamini–Hochberg 보정
- 해석 범위: 제공된 공개 형사 판결문 표본의 변화이며 사법부 전체의 인과적 변화가 아님

## 6. `extras` 폴더

`extras`에는 최종 보고서와 PPT를 만들 때 사용한 생성 코드가 참고용으로 들어 있습니다.
분석 재현에는 필요하지 않으며, Codex 작업환경이나 Microsoft Word 같은 별도 환경에
의존할 수 있습니다.

## 7. 공유 시 주의사항

- 원자료는 용량·권리·보안 문제 때문에 별도로 전달하십시오.
- `crime.zip`의 파일명과 내부 CSV 구조가 달라지면 `01_prepare_data.py`를 조정해야 합니다.
- Windows에서는 한글 그림 글꼴로 맑은 고딕을 우선 사용합니다.

