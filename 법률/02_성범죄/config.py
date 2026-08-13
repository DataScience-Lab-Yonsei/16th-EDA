"""경로 설정 (제출본 재구성).

원본 config.py가 유실되어 01·02 스크립트의 임포트 시그니처
(DATA, NEW_ANAL_SEX, OUTPUT, find_preprocessed)에 맞춰 재작성한 파일이다.
경로 상수만 정의하며 분석 로직에는 관여하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 공통 데이터 폴더 (저장소 최상위의 Dataset/)
DATA = ROOT.parent / "Dataset"

# 01이 저장하고 02가 읽는 태깅 산출물
NEW_ANAL_SEX = DATA / "New_anal_sex.csv"

# 기술통계·회귀 결과 CSV 저장 폴더
OUTPUT = ROOT / "results"


def find_preprocessed() -> Path:
    """공통 전처리 코퍼스(EDA_data_Preprocessed.txt) 위치를 찾는다.

    용량 문제로 저장소에는 포함하지 않았다 — Dataset/README.md의 안내에
    따라 내려받아 Dataset/에 두면 된다.
    """
    candidates = [
        DATA / "EDA_data_Preprocessed.txt",
        ROOT / "EDA_data_Preprocessed.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "EDA_data_Preprocessed.txt를 찾을 수 없습니다. "
        "Dataset/README.md의 원자료 안내를 참고하세요."
    )
