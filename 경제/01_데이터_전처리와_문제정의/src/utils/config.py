"""프로젝트 전역 설정: 경로, 스키마, 상수."""
from __future__ import annotations

from pathlib import Path

import polars as pl

# ── 경로 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data"
DATA_PROC = DATA_RAW / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

_MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def _month_key(p: Path) -> tuple[int, int]:
    """'2019-Oct.csv' → (2019, 10). 알파벳순이 아닌 시간순 정렬을 위해."""
    year, mon = p.stem.split("-")
    return int(year), _MONTHS[mon]


# 시간순 정렬: parquet 행 순서가 시간순이면 event_time 프레디킷 푸시다운이
# row-group 통계로 잘 먹고, 이후 시계열 분할도 자연스럽다.
COSMETICS_CSVS = sorted((DATA_RAW / "archive (1)").glob("*.csv"), key=_month_key)
ELECTRONICS_CSV = DATA_RAW / "events.csv"

STORES = ("cosmetics", "electronics")


def parquet_path(store: str) -> Path:
    return DATA_PROC / f"{store}.parquet"


# ── 타임존 ──────────────────────────────────────────────────────────────
# 원본 event_time 은 전부 UTC. 브랜드(runail/irisk/masura)가 러시아 전문 네일
# 브랜드이므로 시간대 해석은 모스크바 기준이 타당하다.
# 러시아는 2014년 서머타임을 폐지했으므로 2019~2021 구간에서 Europe/Moscow 는
# DST 없이 고정 UTC+3 이다 → 시간대 분석에 왜곡이 없다.
TZ_LOCAL = "Europe/Moscow"

# ── 스키마 ──────────────────────────────────────────────────────────────
# 두 스토어의 스키마를 동일하게 맞춘다(electronics 에는 remove_from_cart 가
# 없지만 Enum 레벨은 공유해야 비교 시 스키마가 갈라지지 않는다).
EVENT_TYPES = ["view", "cart", "remove_from_cart", "purchase"]
EVENT_TYPE_ENUM = pl.Enum(EVENT_TYPES)

# category_id 와 user_id 는 ~1.5e18 까지 관측되므로 Int64 필수.
RAW_SCHEMA: dict[str, pl.DataType] = {
    "event_time": pl.String,  # "2019-10-01 00:00:00 UTC" → 파싱은 loader 에서
    "event_type": pl.String,
    "product_id": pl.Int64,
    "category_id": pl.Int64,
    "category_code": pl.String,
    "brand": pl.String,
    "price": pl.Float64,
    "user_id": pl.Int64,
    "user_session": pl.String,
}

EVENT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # " UTC" 접미사 제거 후 적용
