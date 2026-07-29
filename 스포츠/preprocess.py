"""
StatsBomb Open Data preprocessing for the EDA project:
"축구에서 좋은 공격은 어떻게 만들어지는가?"

Implements 분석 순서 (section 11), Step 1 - Step 6.
Scope: FIFA World Cup 2022 (competition_id=43, season_id=106) - full 360 coverage.

Output: one row per (non-penalty) shot, with attack-sequence variables (9.1)
and defensive-pressure variables (9.2), plus a rule-based attack-type label (10).

Coordinate system (StatsBomb): pitch 120 x 80 (units = yards).
The team in possession always attacks toward x=120. Opponent goal centre = (120, 40),
goalposts at (120, 36) and (120, 44). Penalty box: 102<=x<=120, 18<=y<=62.
Distances are converted yards -> metres (x 0.9144) so the 1/2/3 m thresholds
in the project match physical metres.
"""

import json, os, math
import pandas as pd

CACHE = "cache"
YARD_TO_M = 0.9144
GOAL = (120.0, 40.0)
POST_L = (120.0, 36.0)
POST_R = (120.0, 44.0)

# ---- attack-type thresholds (adjustable) ----
COUNTER_MAX_SEC   = 12.0   # fast counter must be quick
COUNTER_MIN_PROG  = 40.0   # ...and gain lots of ground (x units)
HIGH_TO_MIN_X     = 80.0   # possession won in attacking third
HIGH_TO_MAX_SEC   = 10.0
SLOW_MIN_PASS     = 8
SLOW_MIN_SEC      = 20.0

# ---------------------------------------------------------------- geometry
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def in_box(loc):
    x, y = loc
    return (102.0 <= x <= 120.0) and (18.0 <= y <= 62.0)

def _sign(p1, p2, p3):
    return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])

def in_triangle(pt, v1, v2, v3):
    d1, d2, d3 = _sign(pt, v1, v2), _sign(pt, v2, v3), _sign(pt, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def zone(loc):
    """central / half-space / wide by lateral position."""
    if loc is None:
        return None
    y = loc[1]
    if y < 18 or y > 62:  return "wide"
    if y < 30 or y > 50:  return "half_space"
    return "central"

def abs_sec(e):
    return e.get("minute", 0)*60 + e.get("second", 0)

# ---------------------------------------------------------------- Step 5 features
def freeze_features(shot_loc, freeze_frame):
    """Defensive-pressure variables (9.2) from the shot's freeze frame."""
    if not freeze_frame:
        return {}
    opps = [p for p in freeze_frame if not p.get("teammate", False)]
    mates = [p for p in freeze_frame if p.get("teammate", False)]
    keeper = next((p for p in opps
                   if p.get("position", {}).get("name") == "Goalkeeper"), None)
    dists_m = [dist(shot_loc, p["location"])*YARD_TO_M for p in opps]
    def within(m):  # opponents within m metres of the shooter
        return sum(1 for d in dists_m if d <= m)
    between = sum(1 for p in opps
                  if in_triangle(p["location"], shot_loc, POST_L, POST_R))
    return {
        "n_defenders_ff":        len(opps),
        "nearest_def_dist_m":    round(min(dists_m), 2) if dists_m else None,
        "def_within_1m":         within(1.0),
        "def_within_2m":         within(2.0),
        "def_within_3m":         within(3.0),
        "def_in_box":            sum(1 for p in opps if in_box(p["location"])),
        "def_between_ball_goal": between,
        "teammates_in_box":      sum(1 for p in mates if in_box(p["location"])),
        "gk_x":  round(keeper["location"][0], 1) if keeper else None,
        "gk_y":  round(keeper["location"][1], 1) if keeper else None,
        "gk_dist_from_goalline_m": round((120-keeper["location"][0])*YARD_TO_M, 2) if keeper else None,
        "gk_lateral_offset_m":     round(abs(keeper["location"][1]-40)*YARD_TO_M, 2) if keeper else None,
    }

# ---------------------------------------------------------------- last pass
def classify_last_pass(pass_ev):
    if pass_ev is None:
        return None, None
    pa = pass_ev.get("pass", {})
    loc = pass_ev.get("location")
    end = pa.get("end_location")
    tech = pa.get("technique", {}).get("name")
    if pa.get("cross"):
        ptype = "cross"
    elif pa.get("cut_back"):
        ptype = "cutback"
    elif tech == "Through Ball":
        ptype = "through_ball"
    elif loc and end and loc[0] >= 102 and (end[0] < loc[0]) and abs(end[1]-40) < abs(loc[1]-40):
        # geometric cutback: from near byline, played backward & inward
        ptype = "cutback"
    elif zone(loc) in ("central", "half_space"):
        ptype = "central_pass"
    else:
        ptype = "other"
    return ptype, loc

# ---------------------------------------------------------------- attack type (10)
RECOVERY_TYPES = {"Ball Recovery", "Interception", "Duel"}

def classify_attack(row, start_type, start_x, play_pattern, shot_type):
    if shot_type in ("Free Kick", "Corner"):
        return "set_piece"
    dur, nprog = row["attack_duration_s"], row["x_progression"]
    npass = row["n_pass"]
    if play_pattern == "From Counter" or (dur is not None and dur <= COUNTER_MAX_SEC
                                          and nprog is not None and nprog >= COUNTER_MIN_PROG):
        return "fast_counter"
    if (start_type in RECOVERY_TYPES and start_x is not None and start_x >= HIGH_TO_MIN_X
            and dur is not None and dur <= HIGH_TO_MAX_SEC):
        return "high_turnover"
    lp = row["last_pass_type"]
    if lp == "cross":        return "wide_cross"
    if lp == "cutback":      return "cutback"
    if lp in ("through_ball", "central_pass"):
        return "central_penetration"
    if npass >= SLOW_MIN_PASS and dur is not None and dur >= SLOW_MIN_SEC:
        return "slow_possession"
    return "other"

# ---------------------------------------------------------------- main per-match
def process_match(mid, match_meta, tsx_index):
    events = json.load(open(f"{CACHE}/events/{mid}.json"))
    by_id = {e["id"]: e for e in events}
    rows = []
    shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]
    for s in shots:
        sh = s.get("shot", {})
        stype = sh.get("type", {}).get("name")
        if stype == "Penalty":              # Step 3: exclude penalty kicks
            continue
        loc = s.get("location")
        if not loc:
            continue

        # --- Step 4: attack sequence (possession phase up to the shot) ---
        poss = s.get("possession")
        team_id = s.get("team", {}).get("id")
        seq = [e for e in events if e.get("possession") == poss
               and e.get("index", 0) <= s.get("index", 0)]
        team_seq = [e for e in seq if e.get("team", {}).get("id") == team_id]
        first = next((e for e in team_seq if e.get("location")), None)
        start_loc = first.get("location") if first else None
        start_x = start_loc[0] if start_loc else None
        start_type = first.get("type", {}).get("name") if first else None
        duration = round(abs_sec(s) - abs_sec(first), 1) if first else None
        pre = [e for e in team_seq if e.get("index", 0) < s.get("index", 0)]
        n_pass  = sum(1 for e in pre if e.get("type", {}).get("name") == "Pass"
                      and e.get("pass", {}).get("outcome") is None)  # completed passes
        n_carry = sum(1 for e in pre if e.get("type", {}).get("name") in ("Carry", "Dribble"))
        x_prog  = round(loc[0]-start_x, 1) if start_x is not None else None
        goal_gain = round(dist(start_loc, GOAL)-dist(loc, GOAL), 1) if start_loc else None
        speed = round(x_prog/duration, 2) if (x_prog is not None and duration and duration > 0) else None

        # last pass = the key pass that set up the shot
        kp = by_id.get(sh.get("key_pass_id"))
        last_type, last_loc = classify_last_pass(kp)

        # --- Step 5: freeze frame / 360 pressure ---
        ff = freeze_features(loc, sh.get("freeze_frame"))
        has_360 = s["id"] in tsx_index
        n_tracked = tsx_index.get(s["id"], 0)

        row = {
            "shot_id": s["id"],
            "match_id": mid,
            "team": s.get("team", {}).get("name"),
            "opponent": (match_meta["away"] if s.get("team", {}).get("name") == match_meta["home"]
                         else match_meta["home"]),
            "player": s.get("player", {}).get("name"),
            "period": s.get("period"),
            "minute": s.get("minute"),
            "second": s.get("second"),
            "shot_x": loc[0], "shot_y": loc[1],
            "shot_dist_to_goal_m": round(dist(loc, GOAL)*YARD_TO_M, 2),
            "xg": sh.get("statsbomb_xg"),
            "outcome": sh.get("outcome", {}).get("name"),
            "is_goal": int(sh.get("outcome", {}).get("name") == "Goal"),
            "shot_type": stype,
            "body_part": sh.get("body_part", {}).get("name"),
            "first_time": bool(sh.get("first_time", False)),
            "under_pressure": bool(s.get("under_pressure", False)),
            "play_pattern": s.get("play_pattern", {}).get("name"),
            # 9.1 attack-development variables
            "attack_start_x": start_x,
            "attack_start_y": round(start_loc[1], 1) if start_loc else None,
            "attack_start_type": start_type,
            "attack_duration_s": duration,
            "n_pass": n_pass,
            "n_carry": n_carry,
            "x_progression": x_prog,
            "goal_dist_gain": goal_gain,
            "progression_speed": speed,
            "last_pass_type": last_type,
            "last_pass_x": round(last_loc[0], 1) if last_loc else None,
            "last_pass_y": round(last_loc[1], 1) if last_loc else None,
            "attack_path": zone(last_loc) or zone(loc),
            # 360 linkage
            "has_360_frame": has_360,
            "n_tracked_360": n_tracked,
        }
        row.update(ff)
        row["attack_type"] = classify_attack(row, start_type, start_x,
                                             row["play_pattern"], stype)
        rows.append(row)
    return rows

# ---------------------------------------------------------------- run
def main():
    matches = json.load(open(f"{CACHE}/matches_wc2022.json"))
    all_rows = []
    for m in matches:
        mid = m["match_id"]
        meta = {"home": m["home_team"]["home_team_name"],
                "away": m["away_team"]["away_team_name"]}
        # Step 5: build 360 index (event_uuid -> #tracked players)
        tsx_index = {}
        p360 = f"{CACHE}/360/{mid}.json"
        if os.path.exists(p360):
            for fr in json.load(open(p360)):
                tsx_index[fr["event_uuid"]] = len(fr.get("freeze_frame", []))
        all_rows += process_match(mid, meta, tsx_index)

    df = pd.DataFrame(all_rows)
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)
    df.to_csv("/mnt/user-data/outputs/wc2022_shots.csv", index=False)
    df.to_parquet("/mnt/user-data/outputs/wc2022_shots.parquet", index=False)
    print("rows (non-penalty shots):", len(df))
    print("matches:", df.match_id.nunique())
    print("cols:", len(df.columns))
    print("\nattack_type counts:\n", df.attack_type.value_counts())
    print("\n360 coverage:", df.has_360_frame.mean().round(3))
    print("freeze-frame present:", df.nearest_def_dist_m.notna().mean().round(3))
    print("\nmean xG by attack_type:\n",
          df.groupby("attack_type")["xg"].agg(["count", "mean"]).round(3))
    return df

if __name__ == "__main__":
    main()
