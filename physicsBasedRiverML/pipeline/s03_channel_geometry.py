"""Data preparation 3 of 3: the river's geometry, from the centerline network and the survey itself.

  1. Main-stem centerline: the NHD flowlines around the reach include tributaries and island channels; their parts are
     merged into one continuous line by following, at each junction, the branch nearest the most survey points.
  2. Bank lines and channel polygon: bank points noted in the field are placed on their transects with a linear fit
     through each transect's in-water points; where no bank was surveyed, the edge is the outermost survey point plus
     the median margin measured where both exist; both edges are smoothed over 91 m, kept at least 0.6 m outside the
     survey points, and pushed out until every survey point lies inside the polygon (riverml/geometry.py).

The in-water points that place the edges are the set used when the published bank lines were made (see
legacy_edge_points below); this keeps every number in the paper reproducible.

Inputs:  data/Huron_RiverSystem.geojson, data/25_ and 26_ecohydroData.csv, data/25_ADCPData.csv,
         outputs/intermediate/readings.csv
Outputs: outputs/intermediate/centerline.geojson, bank_lines.geojson, channel.geojson, channel_summary.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import geopandas as gpd
from riverml.config import WORK, RIVER_NETWORK, SURVEY_FILES, ADCP_FILE, ADCP_E, ADCP_N, NORTHING_BAND, FT
from riverml.geometry import CRS, main_stem, bank_lines, BANK_NOTES


def survey_rows():
    """All survey rows assigned to the study reach ('Middle') by the field lead's rule: rows in the northing band, rows with
    no section label, and rows labelled Middle that fall outside the cut-offs of the other two surveyed reaches
    (Upper above 340,000 ft, Lower below 255,000 ft)."""
    raw = pd.concat([pd.read_csv(f) for f in SURVEY_FILES.values()], ignore_index=True)
    raw.loc[raw.Northing > 340000, "Section"] = "Upper"
    raw.loc[(raw.Northing <= NORTHING_BAND[1]) & (raw.Northing >= NORTHING_BAND[0]), "Section"] = "Middle"
    raw.loc[raw.Northing < 255000, "Section"] = "Lower"
    raw["Section"] = raw["Section"].replace("", np.nan).fillna("Middle")
    m = raw[raw.Section == "Middle"].copy()
    m["transect"], m["date"] = m["Transect ID"], pd.to_datetime(m["Collection Date"], format="mixed").dt.date.astype(str)
    return m


def legacy_edge_points(m):
    """The in-water points behind the published bank lines, reproduced from the original data-sorting notebook:
    survey visits (date + point) in water at least 1 ft deep that carry a reading whose recorded height field equals
    1 ft (for 2025 the notebook read the reading-division column), plus the ADCP's 5 h median records at its own
    position, within the 321,000 ft northing limit the notebook applied to this reach. The original notebook is on the
    main branch of github.com/kLabUM/ecologyThresholds_gp (dataSorting.ipynb). Only the bin means along the reach
    depend on which points are used; the edges themselves are set by the bank points and the outermost points."""
    m = m.copy()
    m["h_field"] = np.where(m["Reading Division"].notna(), m["Reading Division"], m["Depth"])
    a = pd.read_csv(ADCP_FILE, skiprows=3)
    a["_time"] = pd.to_datetime(a["_time"])
    a5 = a.set_index("_time")["_value"].resample("300min").median().dropna()
    a5 = a5[np.abs(a5 - a5.mean()) <= 2 * a5.std()]
    adcp = pd.DataFrame({"Collection Date": a5.index.tz_localize(None).astype(str), "Velocity": a5.values / FT, "h_field": 1.0,
                         "Easting": ADCP_E, "Northing": ADCP_N, "Type": "ADCP"})
    w = pd.concat([m, adcp], ignore_index=True)
    w = w.dropna(subset=["Velocity", "h_field"])
    w = w[(w.h_field > 0) & (w.Velocity >= 0)]
    when = pd.to_datetime(w["Collection Date"], format="mixed")
    pid = w["Measurement Point ID"]
    has_pid = pid.notna() & pid.astype(str).str.strip().ne("") & pid.astype(str).ne("nan") & w.Type.ne("ADCP")
    w["visit"] = (when.dt.strftime("%Y-%m-%d") + "|" + pid.astype(str)).where(has_pid, when.astype(str) + "|" + w.index.astype(str))
    keep = []
    for _, g in w.groupby("visit"):
        depth = pd.to_numeric(g["Total water depth"], errors="coerce").dropna()
        if len(depth) and np.median(depth) < 1.0:
            continue
        at_1ft = g[np.isclose(g.h_field, 1.0)]
        if len(at_1ft):
            keep.append(at_1ft.iloc[0])
    k = pd.DataFrame(keep)
    k = k[k.Northing <= 321000]
    k["transect"] = k["Transect ID"]
    k["date"] = pd.to_datetime(k["Collection Date"], format="mixed").dt.date.astype(str)
    return k[["Easting", "Northing", "transect", "date"]]


# ---------------------------------------------------------------- 1. main stem
rd = pd.read_csv(WORK / "readings.csv").drop_duplicates("location_id")
line = main_stem(RIVER_NETWORK, rd.Easting.values, rd.Northing.values)
print(f"main-stem centerline: {line.length * FT:.0f} m long")

# ---------------------------------------------------------------- 2. banks and channel polygon
m = survey_rows()
note = m["Point Notes"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
is_bank = np.zeros(len(m), bool)
for wording in BANK_NOTES:
    is_bank |= note.str.contains(wording, regex=False).values
banks = (m[is_bank].dropna(subset=["Easting", "Northing", "Elevation", "Transect ID"])
         .drop_duplicates(["Transect ID", "Easting", "Northing"]))
left, right, poly, summary = bank_lines(line, legacy_edge_points(m), banks)
print("channel:", json.dumps(summary))

gpd.GeoDataFrame({"name": ["main stem"]}, geometry=[line], crs=CRS).to_file(WORK / "centerline.geojson", driver="GeoJSON")
gpd.GeoDataFrame({"bank": ["left", "right"]}, geometry=[left, right], crs=CRS).to_file(WORK / "bank_lines.geojson", driver="GeoJSON")
gpd.GeoDataFrame({"name": ["wetted channel"]}, geometry=[poly], crs=CRS).to_file(WORK / "channel.geojson", driver="GeoJSON")
json.dump(summary, open(WORK / "channel_summary.json", "w"), indent=1)
