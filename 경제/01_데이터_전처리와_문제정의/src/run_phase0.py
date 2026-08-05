"""Phase 0 실행: CSV → Parquet 변환 후 품질 리포트 생성."""
from __future__ import annotations

import argparse
import time

from src.utils.config import OUTPUTS, STORES, parquet_path
from src.utils.loader import convert_to_parquet
from src.utils.quality import profile, render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for store in STORES:
        t0 = time.time()
        path = convert_to_parquet(store, overwrite=args.overwrite)
        mb = path.stat().st_size / 1e6
        print(f"[parquet] {store:12s} {mb:8.1f} MB  ({time.time() - t0:.1f}s)  {path}")

    profiles = []
    for store in STORES:
        t0 = time.time()
        profiles.append(profile(store))
        print(f"[profile] {store:12s} ({time.time() - t0:.1f}s)")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "00_data_quality_report.md"
    out.write_text(render(profiles), encoding="utf-8")
    print(f"[report ] {out}")


if __name__ == "__main__":
    main()
