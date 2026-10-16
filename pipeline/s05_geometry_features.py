"""Inputs for the Gaussian process (step 3): channel-geometry features of every surveyed location.

  along_m               distance along the main-stem centerline from the ADCP (m, positive downstream)
  rel_across            relative position across the channel, -1 at the right bank to +1 at the left bank
  dist_bank_m           distance to the nearest bank line (m)
  rel_depth             depth at the reference stage relative to the deepest point of the section
  section_mean_depth_m  mean depth of the section at the reference stage (m)
  lateral_bed_slope_abs |d bed / d across| between neighbouring points of a section
Depths are computed at a common reference stage (0.90 m) so they describe the channel's shape, not the flow on the
survey day. Field notes give a substrate class and a wake flag (woody debris, eddies, obstructions) by keyword.

Inputs:  outputs/intermediate/locations.csv, bank_lines.geojson
Output:  outputs/intermediate/features.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from riverml.config import WORK, FT, STAGE_REF, FEATURES

loc = pd.read_csv(WORK / "locations.csv")
loc = loc[loc.a > 0].copy()                                          # still-water locations carry no amplitude

# ---- banks: distance to each bank line and relative position across the channel
bk = gpd.read_file(WORK / "bank_lines.geojson")
left, right = bk[bk.bank == "left"].geometry.iloc[0], bk[bk.bank == "right"].geometry.iloc[0]
pts = [Point(e, n) for e, n in zip(loc.Easting, loc.Northing)]
loc["dist_left_bank_m"] = [left.distance(p) * FT for p in pts]
loc["dist_right_bank_m"] = [right.distance(p) * FT for p in pts]
loc["dist_bank_m"] = loc[["dist_left_bank_m", "dist_right_bank_m"]].min(axis=1)
loc["rel_across"] = (loc.dist_right_bank_m - loc.dist_left_bank_m) / (loc.dist_left_bank_m + loc.dist_right_bank_m)

# ---- depth at the reference stage, relative depth and mean depth of each section, lateral bed slope
loc["h_ref_m"] = np.clip(STAGE_REF - loc.bed_m, 0, None)
by_section = loc.groupby("transect")
loc["rel_depth"] = loc.h_ref_m / by_section.h_ref_m.transform("max")
loc["section_mean_depth_m"] = by_section.h_ref_m.transform("mean")
# gradient of bed elevation against across-channel position, taking a section's points in their stored order
# (survey date, then distance along, then across), as in the published analysis
loc["lateral_bed_slope_abs"] = by_section.apply(
    lambda d: pd.Series(np.gradient(d.bed_m.values, d.across_m.values) if len(d) > 2 else np.zeros(len(d)), index=d.index),
    include_groups=False).reset_index(level=0, drop=True).abs()

# ---- field notes: substrate class (first keyword found) and a wake flag
notes = loc.notes.fillna("").str.lower()
sub = pd.Series("not noted", index=loc.index)
for word, lab in [("cobble", "cobble"), ("gravel", "gravel"), ("sand", "sand"), ("silt", "silt"), ("clay", "silt"),
                  ("veg", "vegetation"), ("grass", "vegetation"), ("wood", "woody debris"), ("debris", "woody debris")]:
    sub[notes.str.contains(word) & (sub == "not noted")] = lab
loc["substrate"] = sub
loc["wake_flag"] = notes.str.contains("debris|wood|eddy|behind|slow|log|branch").astype(int)

loc = loc.sort_values(["transect", "across_m"])
loc[["location_id", "transect", "date", "year", "a_sd", "across_m", "bed_m", "h_ref_m"] + FEATURES + ["substrate", "wake_flag"]].to_csv(
    WORK / "features.csv", index=False)
print(f"{len(loc)} locations; substrate noted at {int((loc.substrate != 'not noted').sum())}, wake flag at {int(loc.wake_flag.sum())}")
