"""정제된 학습 데이터셋 생성 — cart_line + features + target, user_id 해시화.

`features.py` 산출물은 중간 계산용 원시 컬럼(다른 윈도우 타겟, censored_*,
u_first_seen 등)까지 다 담고 있어 그대로 학습에 넣기엔 지저분하다. 이 모듈은
그중 실제 모델 입력만 골라내고(§FEATURE_COLS + 기본 타겟), 관측 기간 부족으로
라벨을 신뢰할 수 없는 ``censored_{DEFAULT_WINDOW}`` 행을 제외한다.

raw ``user_id`` 는 그대로 저장하지 않는다. 완전히 버리면 동일 유저의
cart_line 을 유저 단위로 묶는 것(유저별 GroupKFold, 유저 단위 분석 등)이
불가능해지므로, salted keyed-hash 로 치환한다 — 같은 user_id 는 항상 같은
``user_hash`` 로 사상되어 **유저 단위 이력은 보존**되지만 원본 ID 는
테이블에 남지 않는다.
"""
from __future__ import annotations

import hashlib

import polars as pl

from src.utils.config import DATA_PROC
from src.utils.features import FEATURE_COLS, build_features, scan_features
from src.utils.loader import convert_to_parquet
from src.utils.sessions import DEFAULT_WINDOW, build_cart_lines, build_events

#: 해시에 섞는 고정 키. 바꾸면 기존 산출물의 user_hash 와 어긋나므로 고정한다.
USER_HASH_SALT = b"eda-ecommerce-user-hash-v1"

TARGET = f"y_{DEFAULT_WINDOW}"
CENSORED = f"censored_{DEFAULT_WINDOW}"

#: 학습 테이블에서 제외할 피처. price_missing/price_vs_prior 는 논의 끝에 제외 결정.
EXCLUDE_FEATURES = {"price_missing", "price_vs_prior"}

#: 최종 학습 테이블 컬럼: 식별 키 + 결과 라벨 + 타겟 + 모델 피처.
DATASET_COLS = ["user_hash", "product_id", "t", "outcome", TARGET] + [
    c for c in FEATURE_COLS if c not in EXCLUDE_FEATURES
]


def dataset_path(store: str):
    return DATA_PROC / f"{store}_train.parquet"


def _user_hash_table(user_ids: pl.Series) -> pl.DataFrame:
    """고유 user_id 별 keyed-hash 대응표.

    전체 행이 아니라 고유값에만 해시를 계산한다 — cart_line 은 수백만 행이어도
    고유 유저 수는 그보다 훨씬 적어 이쪽이 빠르다.
    """
    uniq = user_ids.unique().to_list()
    hashes = [
        hashlib.blake2b(str(uid).encode(), digest_size=8, key=USER_HASH_SALT).hexdigest()
        for uid in uniq
    ]
    return pl.DataFrame({
        "user_id": pl.Series(uniq, dtype=user_ids.dtype),
        "user_hash": hashes,
    })


def build_dataset(store: str, *, overwrite: bool = False):
    """raw CSV → parquet → session30 → cart_line → feature 전체 파이프라인을
    이어서 실행하고, 학습용으로 정제한 뒤 user_id 를 해시로 치환해 저장한다.
    """
    out = dataset_path(store)
    if out.exists() and not overwrite:
        return out

    convert_to_parquet(store)
    build_events(store)
    build_cart_lines(store)
    build_features(store)

    df = (
        scan_features(store)
        .filter(~pl.col(CENSORED))
        .select("user_id", *DATASET_COLS[1:])
        .collect()
    )

    hash_tbl = _user_hash_table(df["user_id"])
    df = df.join(hash_tbl, on="user_id", how="left").select(DATASET_COLS)

    df.write_parquet(out, compression="zstd")
    return out


def scan_dataset(store: str) -> pl.LazyFrame:
    return pl.scan_parquet(dataset_path(store))


def main() -> None:
    store = "cosmetics"  # run_phase23 확정: Phase 2 이후는 cosmetics 집중
    path = build_dataset(store)
    stats = scan_dataset(store).select(
        pl.len().alias("n"), pl.col("user_hash").n_unique().alias("n_users"),
    ).collect().row(0, named=True)
    print(f"[{store}] {path} — {stats['n']:,} rows, {stats['n_users']:,} unique users (hashed)")


if __name__ == "__main__":
    main()
