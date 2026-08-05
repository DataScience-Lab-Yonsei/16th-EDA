"""
route_common.py
여행로그 동선 분석 공통 모듈 — 4권역 로딩·코드북·동선 군집을 한 번에 제공.
build() 한 번 호출로 모든 하위 분석(15~18)이 동일한 군집 결과를 공유한다.
"""
import pandas as pd, numpy as np, os, glob, unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE = os.environ.get("EDA_BASE", os.path.dirname(os.path.abspath(__file__)))
ANCHOR = {21, 22, 23, 24, 9}     # 집/친지집/사무실/숙소/역·터미널
NON_DISCRIM = {11, 12}           # 식당/카페·기타 (모든 여행 공통 → 군집 벡터 제외)
EXCLUDE = ANCHOR | NON_DISCRIM
K = 6                            # 동선 전형 수 (silhouette 기준 선정)

# 대표 방문유형 → 전형명 (군집번호가 아니라 실제 대표유형으로 이름 부여 → 재군집/통합 시에도 안정)
_TYPE2NAME = {
    '자연관광지': '자연 탐방형',
    '역사/유적/종교 시설(문화재, 박물관, 촬영지, 절 등)': '역사·유적 탐방형',
    '테마시설(놀이공원, 워터파크)': '테마파크·가족형',
    '레저/스포츠 관련 시설(스키, 카트, 수상레저)': '레저·액티비티형',
    '지역 축제/행사': '축제·이벤트형',
    '상업지구(거리, 시장, 쇼핑시설)': '도심 쇼핑·문화형',
}
# build()에서 대표유형 기반으로 동적 갱신됨 (초기값은 fallback)
CLUSTER_NAME = {}


# 폴더코드 → 권역 / 연도 (145~148=2023 2차년도, 277~280=2022 1차년도)
_REGION_OF = {'145': '수도권', '146': '동부권', '147': '서부권', '148': '제주',
              '277': '수도권', '278': '동부권', '279': '서부권', '280': '제주'}
_YEAR_OF = {'145': 2023, '146': 2023, '147': 2023, '148': 2023,
            '277': 2022, '278': 2022, '279': 2022, '280': 2022}


def _region_dirs():
    dirs = {}
    for r in sorted(os.listdir(BASE)):
        rn = unicodedata.normalize("NFC", r)
        code = rn[:3]
        if code in _REGION_OF:
            tl = glob.glob(os.path.join(BASE, r, "**", "Training", "**", "TL_csv"), recursive=True)
            if tl:
                dirs[f"{_REGION_OF[code]}_{_YEAR_OF[code]}"] = tl[0]
    return dirs


def load_table(region_dir, prefix):
    pn = unicodedata.normalize("NFC", prefix)
    for f in os.listdir(region_dir):
        if f.endswith(".csv") and unicodedata.normalize("NFC", f).startswith(pn):
            return pd.read_csv(os.path.join(region_dir, f), low_memory=False)
    return None


def load_all(prefix, dirs):
    frames = []
    for key, d in dirs.items():
        df = load_table(d, prefix)
        if df is not None:
            region, year = key.rsplit("_", 1)
            df["REGION"] = region
            df["YEAR"] = int(year)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# 동반자 대표유형 → 그룹
_COMP_GROUP = {1: "가족", 2: "가족", 3: "가족", 4: "가족", 5: "가족", 6: "가족",
               7: "친구", 8: "연인", 9: "동료·모임", 10: "동료·모임", 11: "기타"}


def build():
    dirs = _region_dirs()
    visit = load_all("tn_visit_area_info_방문지정보", dirs)
    master = load_all("tn_traveller_master_여행객", dirs)
    travel = load_all("tn_travel_여행", dirs)
    companion = load_all("tn_companion_info_동반자정보", dirs)
    consume = load_all("tn_activity_consume_his_활동소비내역", dirs)
    cb = load_table(list(dirs.values())[0], "tc_codeb_코드B")

    def cmap(a):
        s = cb[cb.cd_a == a]
        return {int(k): v for k, v in zip(s.cd_b, s.cd_nm)}
    VIS, TCR, MOV = cmap("VIS"), cmap("TCR"), cmap("MOV")

    visit["VIS_CD"] = pd.to_numeric(visit["VISIT_AREA_TYPE_CD"], errors="coerce")
    visit["VIS_NM"] = visit["VIS_CD"].map(VIS)
    visit["VISIT_ORDER"] = pd.to_numeric(visit["VISIT_ORDER"], errors="coerce")

    # --- 군집: 순수 관광활동 구성비 ---
    tour = visit[~visit["VIS_CD"].isin(EXCLUDE) & visit["VIS_CD"].notna()]
    comp = pd.crosstab(tour["TRAVEL_ID"], tour["VIS_NM"])
    comp = comp[comp.sum(axis=1) >= 2]
    comp_pct = comp.div(comp.sum(axis=1), axis=0)
    X = StandardScaler().fit_transform(comp_pct.values)
    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
    comp_pct = comp_pct.copy()
    comp_pct["CL"] = km.labels_

    # --- 전형 이름을 대표 방문유형으로 동적 부여 (군집번호 재배정에도 안정) ---
    _prof = comp_pct.drop(columns=["CL"]).groupby(comp_pct["CL"]).mean()
    global CLUSTER_NAME
    _nm, _used = {}, set()
    for cl in sorted(_prof.index):
        base = _TYPE2NAME.get(_prof.loc[cl].idxmax(), str(_prof.loc[cl].idxmax()))
        nm = base if base not in _used else f"{base}#{cl}"
        _used.add(nm); _nm[cl] = nm
    CLUSTER_NAME = _nm
    comp_pct["CL_NM"] = comp_pct["CL"].map(_nm)

    # --- 여행 단위 프로파일 tdf ---
    tdf = pd.DataFrame({"TRAVEL_ID": comp_pct.index, "CL": comp_pct["CL"].values})
    tdf["CL_NM"] = tdf["CL"].map(_nm)
    g = visit.groupby("TRAVEL_ID")
    tdf = tdf.merge(g["VISIT_ORDER"].max().rename("N_VISITS"), on="TRAVEL_ID", how="left")
    tdf = tdf.merge(pd.to_numeric(visit["RESIDENCE_TIME_MIN"], errors="coerce").groupby(visit["TRAVEL_ID"]).mean().rename("STAY_MIN"), on="TRAVEL_ID", how="left")
    tdf = tdf.merge(pd.to_numeric(visit["DGSTFN"], errors="coerce").groupby(visit["TRAVEL_ID"]).mean().rename("SAT"), on="TRAVEL_ID", how="left")
    tdf = tdf.merge(pd.to_numeric(visit["REVISIT_INTENTION"], errors="coerce").groupby(visit["TRAVEL_ID"]).mean().rename("REVISIT"), on="TRAVEL_ID", how="left")
    tdf = tdf.merge(pd.to_numeric(visit["RCMDTN_INTENTION"], errors="coerce").groupby(visit["TRAVEL_ID"]).mean().rename("RCMD"), on="TRAVEL_ID", how="left")
    consume["AMT"] = pd.to_numeric(consume.get("PAYMENT_AMT_WON"), errors="coerce")
    tdf = tdf.merge(consume.groupby("TRAVEL_ID")["AMT"].sum().rename("SPEND"), on="TRAVEL_ID", how="left")
    tm = travel.merge(master[["TRAVELER_ID", "TRAVEL_STYL_1"]], on="TRAVELER_ID", how="left")
    tdf = tdf.merge(tm[["TRAVEL_ID", "REGION"]].drop_duplicates("TRAVEL_ID"), on="TRAVEL_ID", how="left")

    # 동반자 대표유형/그룹 (companion 없으면 '나홀로')
    comp1 = companion.drop_duplicates("TRAVEL_ID")[["TRAVEL_ID", "REL_CD"]].copy()
    comp1["REL_CD"] = pd.to_numeric(comp1["REL_CD"], errors="coerce")
    comp1["REL_NM"] = comp1["REL_CD"].map(TCR)
    comp1["COMP_GRP"] = comp1["REL_CD"].map(_COMP_GROUP)
    tdf = tdf.merge(comp1[["TRAVEL_ID", "REL_NM", "COMP_GRP"]], on="TRAVEL_ID", how="left")
    tdf["COMP_GRP"] = tdf["COMP_GRP"].fillna("나홀로")

    return dict(visit=visit, tour=tour, travel=travel, master=master,
                companion=companion, consume=consume,
                VIS=VIS, TCR=TCR, MOV=MOV,
                comp_pct=comp_pct, tdf=tdf)


# ===== 세부 유형(FINE) 세분 로직 =====
# 방문 선택 이유(VISIT_CHC_REASON_CD = 코드북 REN) 기반 동기 분류
SEEK = {1, 2, 3, 5, 10}   # 지명도/SNS/미디어/지인추천/교육 → 목적형(찾아감)
CONV = {6, 7, 8, 9}       # 교통/편의시설/가성비/우연 → 편의형(그냥)
RETURN = {4}              # 과거경험 → 재방문

_MALL_KW = (r"몰|아울렛|아웃렛|백화점|더\s?현대|현대백화점|신세계|롯데백화점|갤러리아|AK플라자|AK&|"
            r"스타필드|아이파크|IFC|롯데몰|프리미엄|이마트|코스트코|코엑스|센텀시티|스퀘어|플라자|"
            r"지하상가|이케아|가든파이브|현대시티|아트앤사이언스|트리플\s?스트리트")
_CAFE_KW = r"카페|커피|coffee|스타벅스|투썸|이디야|메가|베이커리|빵|제과|디저트|로스터리|브런치"


def add_intent(visit):
    r = pd.to_numeric(visit["VISIT_CHC_REASON_CD"], errors="coerce")
    out = visit.copy()
    out["INTENT"] = np.select(
        [r.isin(SEEK), r.isin(CONV), r.isin(RETURN)],
        ["목적형", "편의형", "재방문"], default="기타")
    return out


def fine_category(visit):
    """각 방문에 세부 유형(FINE_CAT) 부여. 동기+장소명 결합."""
    v = add_intent(visit)
    cd = pd.to_numeric(v["VISIT_AREA_TYPE_CD"], errors="coerce")
    nm = v["VISIT_AREA_NM"].astype(str)
    cat = v["VIS_NM"].astype("object").copy()
    # 상업지구(4) 3분할
    is4 = cd == 4
    market = nm.str.contains("시장", na=False)
    mall = nm.str.contains(_MALL_KW, case=False, na=False, regex=True)
    cat = cat.mask(is4 & market, "전통시장")
    cat = cat.mask(is4 & mall & ~market, "쇼핑몰·아울렛")
    cat = cat.mask(is4 & ~mall & ~market, "거리·상권")
    # 식당/카페(11): 카페·디저트 / 맛집(목적형 식당) / 일반식당
    is11 = cd == 11
    cafe = nm.str.contains(_CAFE_KW, case=False, na=False, regex=True)
    seek = v["INTENT"] == "목적형"
    cat = cat.mask(is11 & cafe & seek, "카페(목적·핫플)")
    cat = cat.mask(is11 & cafe & ~seek, "카페(근처·일반)")
    cat = cat.mask(is11 & ~cafe & seek, "맛집(원정)")
    cat = cat.mask(is11 & ~cafe & ~seek, "일반식당")
    v["FINE_CAT"] = cat
    return v


# 세분 재군집에 쓸 관광활동 카테고리(일반식당·앵커·기타 제외)
FINE_TOUR = ["자연관광지", "역사/유적/종교 시설(문화재, 박물관, 촬영지, 절 등)",
             "문화 시설(공연장, 영화관, 전시관 등)", "체험 활동 관광지",
             "레저/스포츠 관련 시설(스키, 카트, 수상레저)", "테마시설(놀이공원, 워터파크)",
             "지역 축제/행사", "산책로, 둘레길 등", "상점",
             "전통시장", "쇼핑몰·아울렛", "거리·상권", "카페(목적·핫플)", "맛집(원정)"]
