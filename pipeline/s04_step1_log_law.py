"""Step 1. Map each velocity profile to one number.

Every reading at height z above the bed follows the logarithmic law of the wall,
    v(z) = (u*/kappa) ln(z / z0) = a ln(z / z0),
with ONE roughness length z0 for the whole reach, chosen by a grid search that minimises the total squared error over
all profiles with three or more usable readings. With z0 fixed, each location has one free parameter, the amplitude a,
fitted by least squares; its uncertainty is the spread of the values implied by its individual readings (single-reading
locations take the pooled error of predicting the velocity 1 ft above the bed from a profile's other readings).
The depth-averaged velocity at depth h is then v_bar = a [ln(h / z0) - 1].

This step also places each location in the reach frame (distance along the centerline from the ADCP, offset across it)
and puts the RTK bed elevations on the stage sensor's datum: on each survey day, the surveyed water surface minus the
sensor stage gives one offset per day and one water-surface slope for the reach.

Inputs:  outputs/intermediate/readings.csv, centerline.geojson, data/25_ and 26_ecohydroData.csv (RTK elevations)
Outputs: outputs/intermediate/locations.csv (one row per location), sections.csv (T1-T20, north to south),
         z0_search.csv, rtk_bank_points.csv, log_law.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import geopandas as gpd
from riverml.config import WORK, SURVEY_FILES, FT, ADCP_E, ADCP_N, NORTHING_BAND, Z_MIN, V_MIN, Z_REF, Z0_GRID
from riverml.geometry import reach_coordinates

tr = pd.read_csv(WORK / "readings.csv", parse_dates=["time_utc"])
usable = (tr.height_m >= Z_MIN) & (tr.velocity_ms > V_MIN)         # near-bed and below-resolution readings are not fitted

# ---------------------------------------------------------------- 1. one roughness length for the reach
use = tr[usable]
nper = use.groupby("location_id").size()
fitset = use[use.location_id.isin(nper[nper >= 3].index)]


def total_sse(z0):
    tot = 0.0
    for _, p in fitset.groupby("location_id"):
        w = np.log(p.height_m.values / z0)
        v = p.velocity_ms.values
        a = (v * w).sum() / (w * w).sum()
        tot += ((v - a * w) ** 2).sum()
    return tot


z0_grid = np.logspace(np.log10(Z0_GRID[0]), np.log10(Z0_GRID[1]), Z0_GRID[2])
sse = np.array([total_sse(z) for z in z0_grid])
z0 = float(z0_grid[sse.argmin()])
flat = z0_grid[sse <= sse.min() * 1.05]
pd.DataFrame({"z0_m": z0_grid, "sse_m2_s2": sse}).to_csv(WORK / "z0_search.csv", index=False)
print(f"z0 = {z0 * 1000:.1f} mm (within 5% of the best: {flat.min() * 1000:.1f} to {flat.max() * 1000:.1f} mm), "
      f"from {fitset.location_id.nunique()} profiles")

# ---------------------------------------------------------------- 2. one amplitude per location
rows = []
for lid, p in tr.groupby("location_id"):
    ok = p[(p.height_m >= Z_MIN) & (p.velocity_ms > V_MIN)]
    h = float(p.water_depth_m.iloc[0])
    r = {"location_id": lid, "n_readings": len(p), "n_used": len(ok), "water_depth_m": h}
    if len(ok) == 0:                                                 # every reading at or below the floor: still water
        r.update({"a": 0.0, "a_sd": np.nan, "fit_r2": np.nan, "fit_rmse": np.nan, "v_1ft_sd": np.nan, "v_mean": 0.0})
    else:
        w = np.log(ok.height_m.values / z0)
        v = ok.velocity_ms.values
        a = float((v * w).sum() / (w * w).sum())
        implied = v * np.log(Z_REF / z0) / w                         # each reading's implied velocity at 1 ft
        r.update({"a": a,
                  "a_sd": float((v / w).std(ddof=1)) if len(ok) >= 2 else np.nan,
                  "fit_r2": float(1 - ((v - a * w) ** 2).sum() / ((v - v.mean()) ** 2).sum()) if len(ok) >= 3 and v.std() > 0 else np.nan,
                  "fit_rmse": float(np.sqrt(np.mean((v - a * w) ** 2))) if len(ok) >= 3 else np.nan,
                  "v_1ft_sd": float(implied.std(ddof=1)) if len(ok) >= 2 else np.nan,
                  "v_mean": a * (np.log(h / z0) - 1) if h > z0 * np.e else 0.0})
    # transfer check: the velocity measured within 3 cm of 1 ft against the value predicted from the other readings
    near = p[(p.height_m - Z_REF).abs() <= 0.03]
    r["v_measured_near_1ft"] = float(near.velocity_ms.mean()) if len(near) else np.nan
    oth = ok[(ok.height_m - Z_REF).abs() > 0.03]
    if len(near) and len(ok) > len(near) and len(oth):
        w2 = np.log(oth.height_m.values / z0)
        r["v_1ft_from_others"] = float((oth.velocity_ms.values * w2).sum() / (w2 * w2).sum()) * np.log(Z_REF / z0)
    else:
        r["v_1ft_from_others"] = np.nan
    first = p.iloc[0]
    r.update({"year": int(first.year), "date": first.date, "time_utc": p.time_utc.min(), "transect": first.transect,
              "point": first.point, "Easting": first.Easting, "Northing": first.Northing, "x_m": first.x_m, "y_m": first.y_m,
              "stage_m": float(p.stage_m.median()), "notes": first.notes})
    rows.append(r)
loc = pd.DataFrame(rows)
tc = loc.dropna(subset=["v_1ft_from_others", "v_measured_near_1ft"])
transfer_mae = float((tc.v_1ft_from_others - tc.v_measured_near_1ft).abs().mean())
loc["a_sd"] = loc.a_sd.fillna(transfer_mae / np.log(Z_REF / z0))   # single-reading locations: pooled transfer error
loc["bed_elev_m"] = loc.stage_m - loc.water_depth_m                  # bed on the sensor datum from stage and depth
loc["shallower_than_1ft"] = loc.water_depth_m < Z_REF

# ---------------------------------------------------------------- 3. reach frame: along the centerline and across it
line = gpd.read_file(WORK / "centerline.geojson").geometry.iloc[0]
loc["along_m"], loc["across_m"] = reach_coordinates(line, loc.Easting, loc.Northing, origin=(ADCP_E, ADCP_N))

# ---------------------------------------------------------------- 4. RTK bed elevations on the stage datum
raw = pd.concat([pd.read_csv(f, dtype=str) for f in SURVEY_FILES.values()], ignore_index=True)
for c in ["Elevation", "Northing", "Easting"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
for c in ["Transect ID", "Measurement Point ID", "Type"]:
    raw[c] = raw[c].astype(str).str.strip()
raw["date"] = pd.to_datetime(raw["Collection Date"]).dt.strftime("%Y-%m-%d")
for pt in ["AT0710", "AT0711", "AT0714"]:                            # the relabelled points (see s02)
    raw.loc[(raw.date == "2025-07-14") & (raw["Measurement Point ID"] == pt), "Transect ID"] = "AT07"
mid = raw[(raw.Northing > NORTHING_BAND[0]) & (raw.Northing < NORTHING_BAND[1])]
pts = mid[mid.Type == "Transect"].drop_duplicates(["date", "Transect ID", "Measurement Point ID"]).copy()
pts.index = pts.date + "|" + pts["Transect ID"] + "|" + pts["Measurement Point ID"]
loc["bed_rtk_m"] = loc.location_id.map(pts.Elevation) * FT
loc["wse_rtk_m"] = loc.bed_rtk_m + loc.water_depth_m                 # surveyed water surface
ses = loc.groupby(["date", "transect"]).agg(along_m=("along_m", "median"), stage_m=("stage_m", "median"),
                                            wse_m=("wse_rtk_m", "median")).reset_index()
ses["wse_minus_stage_m"] = ses.wse_m - ses.stage_m
ses = ses.sort_values("along_m")
copied = (ses.date == "2025-07-14") & (ses.transect == "AT07")       # these points carry copied elevations, no RTK of their own
fit = ses[~copied]
days = sorted(fit.date.unique())
X = np.zeros((len(fit), len(days) + 1))                              # WSE - stage = offset(day) - slope * along
for k, (_, r) in enumerate(fit.iterrows()):
    X[k, days.index(r.date)] = 1.0
    X[k, -1] = -r.along_m
coef, *_ = np.linalg.lstsq(X, fit.wse_minus_stage_m.values, rcond=None)
slope, offset = float(coef[-1]), pd.Series(coef[:-1], index=days)
resid = fit.wse_minus_stage_m.values - X @ coef
loc["bed_rtk_sensor_datum_m"] = loc.bed_rtk_m - loc.date.map(offset) + slope * loc.along_m
loc.loc[(loc.date == "2025-07-14") & (loc.transect == "AT07"), ["bed_rtk_m", "wse_rtk_m", "bed_rtk_sensor_datum_m"]] = np.nan
loc["bed_m"] = loc.bed_rtk_sensor_datum_m.fillna(loc.bed_elev_m)
print(f"RTK tie: water-surface slope {slope * 1e4:.1f} cm per 100 m, residual sd {resid.std() * 100:.1f} cm")

banks = mid[mid["Point Notes"].fillna("").str.contains("bank height", case=False)][["date", "Transect ID", "Easting", "Northing", "Elevation"]].copy()
banks["along_m"] = [loc[(loc.date == r.date) & (loc.transect == r["Transect ID"])].along_m.median() for _, r in banks.iterrows()]
banks["bank_sensor_datum_m"] = banks.Elevation * FT - banks.date.map(offset) + slope * banks.along_m
banks.to_csv(WORK / "rtk_bank_points.csv", index=False)

loc = loc.sort_values(["date", "transect", "along_m", "across_m"]).reset_index(drop=True)
loc.to_csv(WORK / "locations.csv", index=False)

# sections T1 (north) to T20 (south), by the median northing of their locations
sec = loc.groupby("transect").agg(y_m=("y_m", "median"), along_m=("along_m", "median"), year=("year", "first"),
                                  n_locations=("location_id", "size")).sort_values("y_m", ascending=False)
sec.insert(0, "section", [f"T{i + 1}" for i in range(len(sec))])
sec.reset_index().to_csv(WORK / "sections.csv", index=False)

# ---------------------------------------------------------------- summary numbers quoted in the paper
m3 = loc[loc.n_used >= 3]
fitted = loc[loc.a > 0]
summary = {"z0_m": z0, "z0_within_5pct_m": [float(flat.min()), float(flat.max())], "profiles_in_z0_fit": int(fitset.location_id.nunique()),
           "readings": int(len(tr)), "readings_not_used": int((~usable).sum()), "readings_used": int(usable.sum()),
           "locations": int(len(loc)), "locations_fitted": int(len(fitted)), "locations_still": int((loc.n_used == 0).sum()),
           "locations_by_year_fitted": {str(y): int(n) for y, n in fitted.groupby("year").size().items()},
           "locations_3plus_readings": int(len(m3)), "locations_single_reading": int((loc.n_used == 1).sum()),
           "sections": int(loc.transect.nunique()), "survey_days": int(loc.date.nunique()),
           "fit_r2_quartiles": [float(x) for x in m3.fit_r2.quantile([.25, .5, .75])],
           "fits_r2_below_0": int((m3.fit_r2 < 0).sum()),
           "fit_rmse_median_r2_below_0": float(m3[m3.fit_r2 < 0].fit_rmse.median()),
           "fit_rmse_median_others": float(m3[m3.fit_r2 >= 0].fit_rmse.median()),
           "transfer_check": {"n_profiles": int(len(tc)), "MAE_m_s": transfer_mae},
           "amplitude_a_median_iqr": [float(fitted.a.median()), float(fitted.a.quantile(.25)), float(fitted.a.quantile(.75))],
           "rtk": {"slope_cm_per_100m": slope * 1e4, "residual_sd_cm": float(resid.std() * 100)}}
json.dump(summary, open(WORK / "log_law.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ["locations", "locations_fitted", "fit_r2_quartiles", "fits_r2_below_0", "amplitude_a_median_iqr"]}))
