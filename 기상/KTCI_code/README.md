# 계절별 데이터 기반 KTCI 분석 코드

2023~2025년 관광·기상 결합자료를 이용해 계절별 데이터 기반 KTCI를 산출하고, 2014년 설문 가중치 adapted benchmark 및 민감도 모형과 비교하는 재현용 코드입니다.

## 저장소 구성

```text
KTCI_Code_Submission/
├─ run_all.py
├─ config.yaml
├─ requirements.txt
├─ src/
│  ├─ run_pipeline.py
│  ├─ compare_directional_no_availability.py
│  ├─ compare_absolute_no_availability.py
│  ├─ compare_three_sensitivity_models.py
│  ├─ generate_submission_comparison.py
│  └─ regenerate_figures.py
├─ tests/
│  └─ test_core.py
├─ data/
│  └─ raw/
└─ outputs/
```

`data/raw`와 `outputs`의 대용량 파일은 Git에 포함하지 않습니다.

## 실행 환경

- Python 3.10 이상
- Windows PowerShell 기준

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell의 실행 정책으로 가상환경 활성화가 차단되면 현재 터미널에서만 다음 명령을 먼저 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 입력자료 준비

원자료는 다음 위치에 둡니다.

```text
data/raw/master_tourism_weather_2023_2025.csv
```

기본 입력경로와 변수명은 `config.yaml`에서 변경할 수 있습니다. 원자료에는 최소한 다음 열이 필요합니다.

```text
date
signguCode
signguNm
visitor_total
S_temp
S_humidity
S_wind
S_precip
S_airquality
```

스트레스 스코어는 0이 양호하고 1이 불리한 방향이며, 결측치를 0으로 대체하지 않습니다.

## 전체 분석 실행

```powershell
python run_all.py
```

실행 순서는 다음과 같습니다.

1. STEP 1~7 기본 파이프라인
2. 양의 Spearman을 0으로 처리하고 관측률을 제외한 민감도 모형
3. Spearman 절댓값은 유지하고 관측률만 제외한 민감도 모형
4. 세 데이터 기반 모형의 공통표본 비교
5. 제출용 4모형 비교표 및 그래프 생성

생성 결과는 `outputs/tables`, `outputs/figures`, `outputs/diagnostics`, `outputs/logs`에 저장됩니다.

## 개별 실행

기본 파이프라인:

```powershell
python src/run_pipeline.py --config config.yaml
```

민감도 분석:

```powershell
python src/compare_directional_no_availability.py
python src/compare_absolute_no_availability.py
python src/compare_three_sensitivity_models.py
python src/generate_submission_comparison.py
```

전체 결과와 그래프 재생성:

```powershell
python src/regenerate_figures.py
```

## 테스트

데이터가 없어도 핵심 계산 함수와 설정 구조를 점검할 수 있습니다.

```powershell
pytest -q
```

원자료를 배치한 뒤 전체 통합 실행을 검증하려면 다음을 실행합니다.

```powershell
python run_all.py
pytest -q
```

## 주요 분석 정의

증거점수:

```text
E(j,s) = sqrt(abs(Spearman(j,s)) × decile_amplitude(j,s) × availability(j,s))
```

계절별 가중치:

```text
w(j,s) = E(j,s) / sum(E(k,s))
```

기상 적합도 점수:

```text
Score = 100 × (1 - stress)
```

최종 데이터 기반 KTCI:

```text
KTCI(data) = sum(weight(j,s) × Score(j,i))
```

## 재현성 및 해석상 주의

- `random_seed`는 `config.yaml`에서 관리합니다.
- 2023~2024년을 학습기간, 2025년을 검증기간으로 사용합니다.
- 함안군은 최소 두 개 스트레스 영역을 확보하지 못하면 KTCI를 산출하지 않습니다.
- 시·구 계층이 함께 존재하므로 관광객을 권역별 절대합으로 집계하지 않습니다.
- 2014 비교지수는 원 KTCI의 완전 재현이 아니라, 현재 점수체계의 공통영역에 설문 기반 계절 가중치를 재정규화한 adapted benchmark입니다.
- 원자료와 생성 산출물은 개인정보·용량·재현성 관리를 위해 Git 저장소에서 제외합니다.

