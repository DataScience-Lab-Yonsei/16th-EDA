#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_weather.py
====================
전국 일별 기상데이터(ASOS / AWS / AirKorea) 정규화 파이프라인.
방법론 문서 v0.1 구현 — 5개 영역(기온·습도·대기질·바람·강수)을 0-1로 정규화.
방향: 0 = 좋음/쾌적,  1 = 나쁨/위험 (위해·스트레스 지수)

처리 순서
    소스별 원본(시간자료) 로드 → 결측 마스킹 → 일별 집계 → 파생(체감온도·THI·PMc)
    → 선행연구 앵커 기반 구간선형 정규화 → 소스별 산출 CSV 저장

★ 안전 ★
    - 서버 원본(/data/InSitu/…)은 읽기 전용으로만 접근한다.
    - 산출물은 --output 폴더에만 쓰며, 소스 폴더 내부이면 실행을 거부한다.

의존성 : pandas, numpy  (필수)
    pip install pandas numpy

사용법
    python normalize_weather.py --source all
    python normalize_weather.py --source asos --output ./out
    python normalize_weather.py --source airkorea --limit-files 2   # 테스트용
"""
from __future__ import annotations
import argparse, os, sys, glob, warnings
import numpy as np
import pandas as pd

# 전부 결측인 날의 nanmax/nanmean 경고 억제 (해당 날은 정상적으로 NaN 유지)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===========================================================================
# 0. 설정 : 경로 · 결측코드 · 유효자료 기준
# ===========================================================================
ROOTS = {
    "asos":     "/data/InSitu/KMA_ASOS/raw",
    "aws":      "/data/InSitu/KMA_AWS/month",
    "airkorea": "/data/InSitu/AirKorea/raw_data",
}
NA_VALUES = {
    "asos":     [-9, -9.0, -99, -99.0, "-9", "-9.0", "-99.0", "-99.00", "-9.00"],
    "aws":      [-99, -99.0, "-99", "-99.0"],
    "airkorea": [-999, -999.0, "-999", ""],
}
MIN_VALID_HOURS = 18   # 하루 24시간 중 유효 관측 최소치 (75%)

# ===========================================================================
# 1. 정규화 앵커 (방법론 문서 v0.1) — (xs 농도/값, ys 스트레스 0-1)
# ===========================================================================
A_TEMP_HEAT = ([26, 32, 35, 38, 46], [0.0, 0.40, 0.55, 0.70, 1.0])          # 체감온도(℃)→스트레스
A_TEMP_COLD = ([-40, -27, -13, 0, 9], [1.0, 0.80, 0.55, 0.30, 0.0])         # 체감온도(℃)→스트레스
A_RH_DRY    = ([0, 10, 20, 40], [1.0, 1.0, 0.6, 0.0])                        # RH%(<40)
A_RH_HUMID  = ([60, 80, 95, 100], [0.0, 0.5, 0.9, 1.0])                      # RH%(>60)
A_THI       = ([68, 75, 80, 83], [0.0, 0.5, 0.8, 1.0])                       # 불쾌지수
A_PM25      = ([0, 15, 35, 75, 150], [0, .25, .5, .75, 1.0])                 # ㎍/㎥
A_PMC       = ([0, 15, 45, 75, 150], [0, .25, .5, .75, 1.0])                 # ㎍/㎥ (조대분획)
A_PM10      = ([0, 30, 80, 150, 300], [0, .25, .5, .75, 1.0])               # ㎍/㎥ (PM2.5 미측정기 fallback)
A_O3        = ([0, .03, .09, .15, .30], [0, .25, .5, .75, 1.0])              # ppm (8h)
A_NO2       = ([0, .03, .06, .20, .40], [0, .25, .5, .75, 1.0])              # ppm
A_SO2       = ([0, .02, .05, .15, .30], [0, .25, .5, .75, 1.0])              # ppm
A_CO        = ([0, 2, 9, 15, 30], [0, .25, .5, .75, 1.0])                    # ppm
A_WIND      = ([3.3, 8.0, 13.9, 17.2, 21, 24.5], [0, .25, .5, .70, .85, 1.0])  # m/s
A_RN_AMT    = ([0, 10, 30, 50, 80, 150], [0, .2, .4, .6, .8, 1.0])           # ㎜/일
A_RN_INT    = ([0, 3, 15, 30, 50], [0, .25, .5, .75, 1.0])                   # ㎜/h


def interp_clip(x, anchors):
    """구간선형 보간 후 [0,1] 클리핑. NaN 은 NaN 유지."""
    xs, ys = anchors
    return np.clip(np.interp(x, xs, ys), 0.0, 1.0)


# ===========================================================================
# 2. 파생 지표 : 체감온도 · 불쾌지수
# ===========================================================================
def wind_chill(T, WS_ms):
    """겨울 풍속냉각 체감온도(기상청/Environment Canada). T<=10℃ & 바람 유효시 적용."""
    T = np.asarray(T, float); WS_ms = np.asarray(WS_ms, float)
    V = np.maximum(WS_ms, 0) * 3.6  # km/h
    wct = 13.12 + 0.6215 * T - 11.37 * np.power(np.maximum(V, 0.1), 0.16) \
        + 0.3965 * T * np.power(np.maximum(V, 0.1), 0.16)
    use = (T <= 10) & (V >= 4.8)
    return np.where(use, wct, T)


def heat_index(T, RH):
    """여름 열지수 체감온도(NWS Rothfusz). T>=27℃ 에서 적용, 그 외 기온 그대로."""
    T = np.asarray(T, float); RH = np.asarray(RH, float)
    Tf = T * 9.0 / 5.0 + 32.0
    HI = (-42.379 + 2.04901523 * Tf + 10.14333127 * RH
          - 0.22475541 * Tf * RH - 6.83783e-3 * Tf**2 - 5.481717e-2 * RH**2
          + 1.22874e-3 * Tf**2 * RH + 8.5282e-4 * Tf * RH**2 - 1.99e-6 * Tf**2 * RH**2)
    HI_c = (HI - 32.0) * 5.0 / 9.0
    return np.where(T >= 27, np.maximum(HI_c, T), T)


def discomfort_index(T, RH):
    """불쾌지수 THI (기상청식). T ℃, RH %."""
    T = np.asarray(T, float); RH = np.asarray(RH, float)
    return 1.8 * T - 0.55 * (1 - RH / 100.0) * (1.8 * T - 26.0) + 32.0


def rh_deviation(rh):
    """상대습도 쾌적대(40-60%) 이탈 스트레스."""
    rh = np.asarray(rh, float)
    dry = interp_clip(rh, A_RH_DRY)
    humid = interp_clip(rh, A_RH_HUMID)
    dev = np.where(rh < 40, dry, np.where(rh > 60, humid, 0.0))
    return np.where(np.isnan(rh), np.nan, dev)


# ===========================================================================
# 3. 기상 소스(ASOS/AWS) : 일별 집계 + 정규화
# ===========================================================================
def _daily_met(df, colmap):
    """시간자료 → (station,date) 일별 집계."""
    dt = df[colmap["datetime"]].astype("int64").astype(str)
    df = df.assign(_date=dt.str[:8], _stn=df[colmap["station"]])
    g = df.groupby(["_stn", "_date"])
    out = pd.DataFrame({
        "TA_mean": g[colmap["TA"]].mean(), "TA_max": g[colmap["TA"]].max(),
        "TA_min":  g[colmap["TA"]].min(),  "n_TA": g[colmap["TA"]].count(),
        "HM_mean": g[colmap["HM"]].mean(), "n_HM": g[colmap["HM"]].count(),
        "WS_mean": g[colmap["WS"]].mean(), "WS_max": g[colmap["WS"]].max(),
        "n_WS": g[colmap["WS"]].count(),
        "RN_sum": g[colmap["precip"]].sum(min_count=1),
        "RN_max": g[colmap["precip"]].max(), "n_RN": g[colmap["precip"]].count(),
    }).reset_index().rename(columns={"_stn": "station", "_date": "date"})
    return out


def _mask_invalid(daily):
    """유효 관측시간 미만 변수는 NaN 처리."""
    m = MIN_VALID_HOURS
    for var, ncol in [("TA_mean", "n_TA"), ("TA_max", "n_TA"), ("TA_min", "n_TA"),
                      ("HM_mean", "n_HM"), ("WS_mean", "n_WS"), ("WS_max", "n_WS")]:
        daily.loc[daily[ncol] < m, var] = np.nan
    # 강수는 결측을 0으로 오인하면 안 되므로 유효시간 부족시 NaN
    daily.loc[daily["n_RN"] < m, ["RN_sum", "RN_max"]] = np.nan
    return daily


def normalize_met(daily):
    """기상 4개 영역 정규화."""
    # 파생
    AT_day = heat_index(daily["TA_max"], daily["HM_mean"])
    AT_night = wind_chill(daily["TA_min"], daily["WS_mean"])
    THI = discomfort_index(daily["TA_mean"], daily["HM_mean"])
    daily["AT_day"] = AT_day
    daily["AT_night"] = AT_night
    daily["THI"] = THI
    # ① 기온 : 열/냉 스트레스의 max
    s_heat = interp_clip(AT_day, A_TEMP_HEAT)
    s_cold = interp_clip(AT_night, A_TEMP_COLD)
    daily["S_temp"] = np.nanmax(np.vstack([s_heat, s_cold]), axis=0)
    # ② 습도 : RH 이탈 / THI 의 max
    s_rh = rh_deviation(daily["HM_mean"])
    s_thi = interp_clip(THI, A_THI)
    daily["S_humidity"] = np.nanmax(np.vstack([s_rh, s_thi]), axis=0)
    # ④ 바람 : 일최대풍속 기준
    daily["S_wind"] = interp_clip(daily["WS_max"], A_WIND)
    # ⑤ 강수 : 강수량 / 강도 의 max (무강수일=0)
    s_amt = interp_clip(daily["RN_sum"], A_RN_AMT)
    s_int = interp_clip(daily["RN_max"], A_RN_INT)
    daily["S_precip"] = np.nanmax(np.vstack([s_amt, s_int]), axis=0)
    return daily


def process_met(source, files, out_path):
    colmaps = {
        "asos": dict(datetime="KST", station="STN", TA="TA", HM="HM", WS="WS", precip="RN"),
        "aws":  dict(datetime="KST", station="STN", TA="TA", HM="HM", WS="WS", precip="RN_HR1"),
    }
    cm = colmaps[source]
    need = list(dict.fromkeys(cm.values()))
    dailies = []
    for i, fp in enumerate(files, 1):
        try:
            df = pd.read_csv(fp, encoding="utf-8-sig", na_values=NA_VALUES[source])
            df.columns = [c.strip() for c in df.columns]
            miss = [c for c in need if c not in df.columns]
            if miss:
                print(f"    (건너뜀) {os.path.basename(fp)} : 컬럼없음 {miss}"); continue
            dailies.append(_daily_met(df[need], cm))
            print(f"    [{i}/{len(files)}] {os.path.basename(fp)}  일수집계 {len(dailies[-1])}")
        except Exception as e:
            print(f"    (오류 건너뜀) {os.path.basename(fp)} : {e}")
    if not dailies:
        print("    처리할 데이터 없음"); return
    daily = pd.concat(dailies, ignore_index=True)
    # 월경계 없이 파일=월 단위라 (station,date) 중복 없음. 안전차원 재집계 생략.
    daily = _mask_invalid(daily)
    daily = normalize_met(daily)
    cols = ["station", "date", "S_temp", "S_humidity", "S_wind", "S_precip",
            "TA_mean", "TA_max", "TA_min", "HM_mean", "WS_mean", "WS_max",
            "RN_sum", "RN_max", "AT_day", "AT_night", "THI",
            "n_TA", "n_HM", "n_WS", "n_RN"]
    daily = daily.sort_values(["station", "date"])[cols].round(4)
    daily.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    ▶ 저장: {out_path}  ({len(daily):,}행, 지점 {daily['station'].nunique()}개)")


# ===========================================================================
# 4. AirKorea : 일별 집계 + 대기질 정규화
# ===========================================================================
AK_COLS7 = ["station", "dt", "SO2", "CO", "O3", "NO2", "PM10"]
AK_COLS8 = ["station", "dt", "SO2", "CO", "O3", "NO2", "PM25", "PM10"]


def _daily_airkorea(df):
    """시간자료 → (station,date) 일별. O3 는 일 최대 8시간 이동평균, 나머지 일평균."""
    df = df.copy()
    df["dt"] = df["dt"].astype("int64").astype(str)
    df["date"] = df["dt"].str[:8]
    df["hour"] = df["dt"].str[8:10]
    gases = [c for c in ["SO2", "CO", "NO2", "PM25", "PM10"] if c in df.columns]
    g = df.groupby(["station", "date"])
    agg = {c: "mean" for c in gases}
    daily = g.agg(agg)
    daily["n_obs"] = g["dt"].count()
    # O3 8시간 이동평균의 일 최대
    df = df.sort_values(["station", "dt"])
    o3_8h = df.groupby("station")["O3"].transform(lambda s: s.rolling(8, min_periods=6).mean())
    df["_O3_8h"] = o3_8h
    o3d = df.groupby(["station", "date"])["_O3_8h"].max()
    daily["O3"] = o3d
    return daily.reset_index()


def normalize_airkorea(daily):
    # 입자상 처리 : PM2.5 있는 행은 PM2.5+PMc, 없는 행(2001-2018)은 PM10 fallback.
    # 행 단위로 분기해 초기연도 미세먼지 누락을 방지한다.
    subs = {}
    if "PM25" in daily.columns:
        pm25 = daily["PM25"]
        pmc = np.maximum(daily["PM10"] - pm25, 0.0)
        daily["PMc"] = pmc
        pm25_missing = pm25.isna().values
        s_pm10_fb = interp_clip(daily["PM10"], A_PM10)
        subs["pm25"] = interp_clip(pm25, A_PM25)
        subs["pmc"]  = interp_clip(pmc, A_PMC)
        # PM2.5 결측 행에서만 PM10 fallback 사용(측정된 행은 NaN 처리해 이중계산 방지)
        subs["pm10"] = np.where(pm25_missing, s_pm10_fb, np.nan)
    else:
        subs["pm10"] = interp_clip(daily["PM10"], A_PM10)
    subs.update({
        "o3":  interp_clip(daily["O3"], A_O3),
        "no2": interp_clip(daily["NO2"], A_NO2),
        "so2": interp_clip(daily["SO2"], A_SO2),
        "co":  interp_clip(daily["CO"], A_CO),
    })
    S = pd.DataFrame(subs, index=daily.index)
    for k in S.columns:
        daily[f"s_{k}"] = S[k].round(4)
    daily["S_airquality"] = np.nanmax(S.values, axis=1).round(4)      # 최악물질
    daily["S_aq_mean"] = np.nanmean(S.values, axis=1).round(4)         # 참고: 평균
    return daily


def process_airkorea(files, out_path):
    dailies = []
    for i, fp in enumerate(files, 1):
        try:
            first = pd.read_csv(fp, header=None, nrows=1)
            ncol = first.shape[1]
            names = AK_COLS8 if ncol == 8 else AK_COLS7
            df = pd.read_csv(fp, header=None, names=names,
                             na_values=NA_VALUES["airkorea"])
            dailies.append(_daily_airkorea(df))
            print(f"    [{i}/{len(files)}] {os.path.basename(fp)} ({ncol}열)  일수집계 {len(dailies[-1])}")
        except Exception as e:
            print(f"    (오류 건너뜀) {os.path.basename(fp)} : {e}")
    if not dailies:
        print("    처리할 데이터 없음"); return
    daily = pd.concat(dailies, ignore_index=True)
    # 분기/월 파일이라 같은 (station,date) 중복 없음 → 재집계 불필요
    daily.loc[daily["n_obs"] < MIN_VALID_HOURS,
              [c for c in ["SO2", "CO", "O3", "NO2", "PM25", "PM10"] if c in daily.columns]] = np.nan
    daily = normalize_airkorea(daily)
    base = ["station", "date", "S_airquality", "S_aq_mean"]
    scols = [c for c in daily.columns if c.startswith("s_")]
    raw = [c for c in ["PM25", "PMc", "PM10", "O3", "NO2", "SO2", "CO", "n_obs"] if c in daily.columns]
    daily = daily.sort_values(["station", "date"])[base + scols + raw].round(4)
    daily.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    ▶ 저장: {out_path}  ({len(daily):,}행, 측정소 {daily['station'].nunique()}개)")


# ===========================================================================
# 5. main
# ===========================================================================
def list_files(source):
    root = ROOTS[source]
    if source == "airkorea":
        return sorted(glob.glob(os.path.join(root, "*", "*.csv")))
    return sorted(glob.glob(os.path.join(root, "*.csv")))


def is_subpath(child, parent):
    try:
        return os.path.commonpath([os.path.realpath(child), os.path.realpath(parent)]) == os.path.realpath(parent)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="기상데이터 5영역 정규화 (0=좋음,1=나쁨)")
    ap.add_argument("--source", choices=["asos", "aws", "airkorea", "all"], default="all")
    ap.add_argument("--output", default="./normalized_output")
    ap.add_argument("--limit-files", type=int, default=0, help="소스별 처리 파일 수 제한(테스트)")
    args = ap.parse_args()

    out_abs = os.path.realpath(args.output)
    for r in ROOTS.values():
        if os.path.exists(r) and is_subpath(out_abs, r):
            print(f"[중단] 출력 폴더가 소스 내부입니다: {out_abs}"); sys.exit(1)
    os.makedirs(out_abs, exist_ok=True)

    sources = ["asos", "aws", "airkorea"] if args.source == "all" else [args.source]
    for src in sources:
        print(f"\n=== {src.upper()} ===")
        files = list_files(src)
        if args.limit_files:
            files = files[:args.limit_files]
        if not files:
            print(f"    파일 없음: {ROOTS[src]}"); continue
        out_path = os.path.join(out_abs, f"{src}_normalized.csv")
        if src == "airkorea":
            process_airkorea(files, out_path)
        else:
            process_met(src, files, out_path)
    print("\n완료.")


if __name__ == "__main__":
    main()
