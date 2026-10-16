"""Data preparation 2 of 3: the field surveys (hand-held velocity readings and smallmouth bass nests).

Each survey row is one reading: a velocity at a height above the bed, at a point on a transect, with the local
water depth and RTK coordinates. Rows within the study reach's northing band are kept, converted to metres and to
a local frame centred on the ADCP; each velocity reading is given the stage at its time (from s01).

Named corrections, each confirmed by the field lead:
  * three 14 Jul 2025 points carry the previous transect's ID but lie on AT07's line: relabelled AT07
  * one reading of 3.60 ft/s at 0.08 ft in a profile otherwise reading 0.5-0.8 ft/s: a stray digit, removed

Inputs:  data/25_ and 26_ecohydroData.csv, outputs/intermediate/stage_adcp_10min.csv
Outputs: outputs/intermediate/readings.csv (one row per velocity reading, with stage), nests.csv (one row per nest record)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from riverml.config import WORK, SURVEY_FILES, FT, ADCP_E, ADCP_N, NORTHING_BAND, TZ_SURVEY

RELABEL = {("7/14/2025", "AT0710"): "AT07", ("7/14/2025", "AT0711"): "AT07", ("7/14/2025", "AT0714"): "AT07"}
ENTRY_ERRORS = {("2025-07-14", "AT06", "AT0605", 0.08)}          # (date, transect, point, height in ft)

s = pd.concat([pd.read_csv(f, dtype=str).assign(year=yr) for yr, f in SURVEY_FILES.items()], ignore_index=True)
for c in ["Total water depth", "Depth", "Velocity", "Northing", "Easting", "Latitude", "Longitude"]:
    s[c] = pd.to_numeric(s[c], errors="coerce")
for c in ["Transect ID", "Measurement Point ID", "Type"]:
    s[c] = s[c].astype(str).str.strip().replace({"nan": np.nan})        # e.g. ' AT11' carried a leading space
for (d, pt), t in RELABEL.items():
    s.loc[(s["Collection Date"].str.strip() == d) & (s["Measurement Point ID"] == pt), "Transect ID"] = t
# rows without a recorded time are placed at noon of their survey day
s["time_local"] = pd.to_datetime(s["Collection Date"].str.strip() + " " + s["Collection Time"].fillna("12:00").str.strip(),
                                 format="mixed", errors="coerce")
s["time_utc"] = s["time_local"].dt.tz_localize(TZ_SURVEY, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
s["date"] = s["time_local"].dt.date.astype(str)

r = s[(s.Northing > NORTHING_BAND[0]) & (s.Northing < NORTHING_BAND[1])].copy()
r["height_ft"] = r["Depth"]                                          # 'Depth' is the reading's height above the bed (ft)
r["height_m"] = r.height_ft * FT
r["water_depth_m"] = r["Total water depth"] * FT
r["velocity_ms"] = r.Velocity * FT
r["x_m"], r["y_m"] = (r.Easting - ADCP_E) * FT, (r.Northing - ADCP_N) * FT   # metres east and north of the ADCP
r["location_id"] = r.date + "|" + r["Transect ID"].astype(str) + "|" + r["Measurement Point ID"].astype(str)
r = r.rename(columns={"Transect ID": "transect", "Measurement Point ID": "point", "Point Notes": "notes", "Type": "type"})
keep = ["year", "date", "time_utc", "type", "transect", "point", "location_id", "Easting", "Northing", "Latitude", "Longitude",
        "x_m", "y_m", "water_depth_m", "height_ft", "height_m", "velocity_ms", "notes"]

# ---------------------------------------------------------------- velocity readings on transects
tr = r[r.type == "Transect"][keep].copy()
n0 = len(tr)
tr = tr[tr.velocity_ms.notna() & tr.height_m.notna() & (tr.velocity_ms >= 0)]
tr = tr[~(tr.height_m > tr.water_depth_m + 0.02)]                   # a reading cannot sit above the water surface
bad = [(d, t, p, round(h, 2)) in ENTRY_ERRORS for d, t, p, h in zip(tr.date, tr.transect, tr.point, tr.height_ft)]
tr = tr[~np.array(bad)]
tr = tr.sort_values(["time_utc", "location_id", "height_m"]).reset_index(drop=True)
nests = r[r.type == "Nest"][keep].copy()

# ---------------------------------------------------------------- stage at the time of each reading
g = pd.read_csv(WORK / "stage_adcp_10min.csv", parse_dates=["time_utc"]).set_index("time_utc")
g.index = pd.to_datetime(g.index, utc=True)
sm = g.stage_m.dropna().sort_index()


def stage_at(t):
    """Linear interpolation in the cleaned stage record; NaN if the nearest value is more than 2 h away."""
    if pd.isna(t):
        return np.nan
    i = sm.index.searchsorted(t)
    lo, hi = max(i - 1, 0), min(i, len(sm) - 1)
    gap = min(abs((sm.index[lo] - t).total_seconds()), abs((sm.index[hi] - t).total_seconds())) / 3600
    if gap > 2:
        return np.nan
    w = 0.5 if sm.index[hi] == sm.index[lo] else (t - sm.index[lo]) / (sm.index[hi] - sm.index[lo])
    return float(sm.iloc[lo] * (1 - w) + sm.iloc[hi] * w)


tr["stage_m"] = [stage_at(t) for t in tr.time_utc]
tr.to_csv(WORK / "readings.csv", index=False)
nests.to_csv(WORK / "nests.csv", index=False)
print(f"{n0} transect rows -> {len(tr)} velocity readings at {tr.location_id.nunique()} locations on {tr.date.nunique()} days; "
      f"{nests.location_id.nunique()} nest records; {int(tr.stage_m.isna().sum())} readings without a stage")
