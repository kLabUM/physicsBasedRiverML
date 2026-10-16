"""Data preparation 1 of 3: the continuous water level (stage) record and the ADCP velocity record.

Stage: the bridge-mounted sensor reports the range to the water surface every 10 min; stage = mounting offset - range.
  1. drop physically implausible readings and spikes that depart from a 6 h rolling median
  2. average onto a 10 min grid
  3. remove the sensor's small daily pattern: a 1 h median filter, a night line (median of the readings taken with the
     sun below the horizon, joined across each day), the mean daytime departure from that line on dry days by month,
     subtracted, then a 3 h and a 2 h rolling median
ADCP: the bed-mounted ADCP's velocity on the same 10 min grid, smoothed with the same median filters.

Inputs:  data/25_ and 26_depthSensorData_Middle.csv, data/25_ADCPData.csv
Output:  outputs/intermediate/stage_adcp_10min.csv  (time_utc, stage_m, stage_raw_m, adcp_ms, adcp_raw_ms)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from riverml.config import WORK, STAGE_FILES, ADCP_FILE, STAGE_DATUM_MM, STAGE_PLAUSIBLE_M, SPIKE_M, SITE_LAT, SITE_LON

# ---------------------------------------------------------------- 1-2. range readings to stage, cleaned, on a 10 min grid
st = []
for f in STAGE_FILES:
    d = pd.read_csv(f, skiprows=3)
    d = d[d["_field"] == "value"]
    st.append(pd.DataFrame({"time_utc": pd.to_datetime(d["_time"], utc=True), "range_mm": d["_value"].astype(float)}))
st = pd.concat(st).sort_values("time_utc").drop_duplicates("time_utc").reset_index(drop=True)
st["stage_m"] = (STAGE_DATUM_MM - st.range_mm) / 1000.0
ok = st.stage_m.between(*STAGE_PLAUSIBLE_M)
st = st[ok].set_index("time_utc")
spike = (st.stage_m - st.stage_m.rolling("6h", center=True).median()).abs() > SPIKE_M
st = st[~spike]
raw10 = st.stage_m.resample("10min").mean().dropna()
print(f"stage: {int((~ok).sum())} implausible readings and {int(spike.sum())} spikes removed; {len(raw10)} ten-minute values")


# ---------------------------------------------------------------- 3. the daily sensor pattern, removed against the night line
def sun_elevation(t_utc, lat=SITE_LAT, lon=SITE_LON):
    """Solar elevation in degrees (NOAA approximation) for a UTC DatetimeIndex."""
    doy = t_utc.dayofyear.values
    hour = (t_utc.hour + t_utc.minute / 60).values
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.deg2rad((hour * 60 + eqt + 4 * lon) / 4 - 180)
    return np.rad2deg(np.arcsin(np.sin(np.deg2rad(lat)) * np.sin(decl) + np.cos(np.deg2rad(lat)) * np.cos(decl) * np.cos(ha)))


df = pd.DataFrame({"raw": raw10})
df["sun"] = sun_elevation(raw10.index)
df = df.tz_convert("Etc/GMT+5")                                                   # UTC-5 (EST, no daylight saving), for hour of day
df["filtered"] = df.raw.rolling(7, center=True, min_periods=3).median()            # 1 h median filter
night = df.sun < 0
night_id = ((~night).astype(int).diff().fillna(0) == -1).cumsum()                # one id per night
nl = df[night].groupby(night_id[night]).filtered.agg(["median"])
nl["t"] = df[night].groupby(night_id[night]).apply(lambda g: g.index[len(g) // 2])
df["night_line"] = np.interp(df.index.astype("int64"), nl.t.astype("int64").values, nl["median"].values)
df["hbin"] = (df.index.hour + df.index.minute / 60) // 0.5 * 0.5
df["month"] = df.index.strftime("%Y-%m")
df["night_date"] = (df.index + pd.Timedelta(hours=1)).normalize()
nm = df[night].groupby("night_date").filtered.median()
df["dry"] = df.night_date.isin(nm[nm.diff().abs() < 0.02].index)                # night-to-night change under 2 cm
df["departure"] = df.filtered - df.night_line
pattern_all = df[df.dry].groupby("hbin").departure.mean()
pattern = {m: (g.groupby("hbin").departure.mean().reindex(pattern_all.index).interpolate() if g.night_date.nunique() >= 6
               else pattern_all) for m, g in df[df.dry].groupby("month")}               # one daily pattern per month
df["pattern"] = [pattern.get(m, pattern_all).get(h, 0.0) for m, h in zip(df.month, df.hbin)]
df["corrected"] = (df.filtered - df.pattern).rolling("3h", center=True).median()   # pattern removed, 3 h median
df = df.tz_convert("UTC")

grid = pd.date_range(df.index.min().floor("10min"), df.index.max().ceil("10min"), freq="10min", tz="UTC")
out = pd.DataFrame(index=grid)
out.index.name = "time_utc"
out["stage_raw_m"] = df.raw.reindex(grid, method="nearest", tolerance=pd.Timedelta("5min"))
out["stage_m"] = (df.corrected.reindex(grid, method="nearest", tolerance=pd.Timedelta("5min"))
                  .rolling(13, center=True, min_periods=5).median())                 # a final 2 h median

# ---------------------------------------------------------------- ADCP on the same grid, same filters
a = pd.read_csv(ADCP_FILE, skiprows=3)
a = a[a["_field"] == "value"]
ad = pd.Series(a["_value"].astype(float).values, index=pd.to_datetime(a["_time"], utc=True)).sort_index()
out["adcp_raw_ms"] = ad.resample("10min").median().reindex(grid)
v = out.adcp_raw_ms.rolling(7, center=True, min_periods=3).median()                  # 1 h median (glitches)
v = v.rolling(19, center=True, min_periods=7).median()                               # 3 h median
v = v.rolling(13, center=True, min_periods=5).median()                               # 2 h median
out["adcp_ms"] = v.rolling(7, center=True, min_periods=3).mean()                     # 1 h mean: the ADCP logs in 0.01 m/s steps
out.loc[out.adcp_raw_ms.isna(), "adcp_ms"] = np.nan
out.to_csv(WORK / "stage_adcp_10min.csv")
print(f"stage record {out.stage_m.first_valid_index():%Y-%m-%d} to {out.stage_m.last_valid_index():%Y-%m-%d}; "
      f"ADCP {int(out.adcp_raw_ms.notna().sum())} ten-minute values")
