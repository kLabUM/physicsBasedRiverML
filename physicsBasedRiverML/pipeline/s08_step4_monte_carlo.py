"""Step 4. Extend through time.

The stage record drives the model through both seasons: every 2 h, 100 joint Monte Carlo draws of the whole reach
(riverml/chain.py) give each section's mean depth-averaged velocity with a 90% band.

Validation with whole survey days held out, one day at a time, the full chain refitted without the day:
  * section level: each held-out section's measured mean velocity against its 90% band;
  * reading level: each held-out reading against the velocity predicted at its own height, with a 95% band.
Attribution: the band at four sections spread along the reach (T4, T11, T16, T20) and four stages, with one source
of uncertainty switched on at a time.

Inputs:  outputs/intermediate/chain.pkl, stage_adcp_10min.csv, readings.csv, sections.csv
Outputs: outputs/intermediate/section_velocity_2h.csv, heldout_sections.csv, heldout_readings.csv,
         uncertainty_attribution.csv, step4.json"""
import sys, json, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from riverml.config import WORK, SURVEYED, N_DRAWS
from riverml.gp import Model, with_target, stage_factor
from riverml.chain import Chain, sample_reach, section_mean, width_weights, ALL_SOURCES
warnings.filterwarnings("ignore")

chain = pickle.load(open(WORK / "chain.pkl", "rb"))
df, Z0 = chain.df, chain.z0
H0, GAMMA = chain.primary
sec_pts = {t: g.assign(_x=g.x_m.round(2), _y=g.y_m.round(2)).drop_duplicates(["_x", "_y"]).sort_values("across_m").reset_index(drop=True)
           for t, g in df.groupby("transect")}                     # one point per position, ordered across the channel
order = sorted(sec_pts, key=lambda t: sec_pts[t].along_m.median())
sections = pd.read_csv(WORK / "sections.csv").set_index("transect").section

# ---------------------------------------------------------------- 1. every section, every 2 h, both seasons
g10 = pd.read_csv(WORK / "stage_adcp_10min.csv", parse_dates=["time_utc"]).set_index("time_utc")
g10.index = pd.to_datetime(g10.index, utc=True)
st2 = g10.stage_m.resample("2h").mean().dropna()
st2 = st2[st2.index.year >= 2025]
R = sample_reach(chain, st2.values, {t: sec_pts[t] for t in order}, n=N_DRAWS, seed=7)
ser = pd.concat([pd.DataFrame({"transect": t, "section": sections[t], "time_utc": st2.index, "stage_m": st2.values,
                               "mean_v_med": np.nanmedian(R[t], 0), "mean_v_lo": np.nanquantile(R[t], .05, 0),
                               "mean_v_hi": np.nanquantile(R[t], .95, 0)}) for t in order], ignore_index=True)
ser.to_csv(WORK / "section_velocity_2h.csv", index=False)
print(f"{len(order)} sections x {len(st2)} two-hour stages")

# ---------------------------------------------------------------- 2. survey days held out: sections and readings
cov_rows, RH = [], []
rd = pd.read_csv(WORK / "readings.csv")
dl = with_target(df, H0, GAMMA)
for day in sorted(df.date.unique()):
    ch_d = Chain(df[df.date != day], chain.ridge, Z0, n_restarts=1)          # the whole chain refitted without the day
    for t in [t for t in order if sec_pts[t].date.iloc[0] == day]:
        p = sec_pts[t]
        stage = float(p.stage_m.median())
        mv = sample_reach(ch_d, [stage], {t: p}, n=300, seed=21)[t][:, 0]
        w = width_weights(p)
        vb = p.a.values * (np.log(np.clip(p.water_depth_m.values, np.e * Z0, None) / Z0) - 1)   # measured v_bar per point
        cov_rows.append({"transect": t, "section": sections[t], "date": day, "stage_m": stage, "mean_v_measured": float(np.sum(vb * w) / w.sum()),
                         "mean_v_med": float(np.nanmedian(mv)), "mean_v_lo": float(np.nanquantile(mv, .05)), "mean_v_hi": float(np.nanquantile(mv, .95))})
    te = dl[dl.date == day]                                               # readings: the six-feature GP without the day
    mu, sd = Model(n_restarts=1).fit(dl[dl.date != day]).predict(te)
    a_hat = dict(zip(te.location_id, np.exp(mu) * stage_factor(te.stage_m.values, H0, GAMMA)))   # a = A (s - h0)^gamma
    s_hat = dict(zip(te.location_id, np.sqrt(sd ** 2 + te.sd_y.values ** 2)))
    r = rd[rd.location_id.isin(te.location_id) & (rd.height_m > Z0 * np.e)].copy()
    shape = np.log(r.height_m.values / Z0)
    r["v_pred"] = [a_hat[l] for l in r.location_id] * shape
    r["v_lo"] = [a_hat[l] * np.exp(-1.96 * s_hat[l]) for l in r.location_id] * shape
    r["v_hi"] = [a_hat[l] * np.exp(1.96 * s_hat[l]) for l in r.location_id] * shape
    RH.append(r)
    print(f"held out {day}")
covs = pd.DataFrame(cov_rows)
covs["in_band"] = (covs.mean_v_measured >= covs.mean_v_lo) & (covs.mean_v_measured <= covs.mean_v_hi)
covs.to_csv(WORK / "heldout_sections.csv", index=False)
RH = pd.concat(RH)
RH.to_csv(WORK / "heldout_readings.csv", index=False)

# ---------------------------------------------------------------- 3. which source sets the band
att = []
for t in ["AT10", "0625-T02", "AT0401", "AT13"]:                      # sections T16, T4, T11, T20
    pts = df[df.transect == t].sort_values("across_m").reset_index(drop=True)
    for st in [0.75, 0.90, 1.10, 1.30]:
        mv = section_mean(chain.sample(st, pts, n=400, seed=1))
        row = {"transect": t, "section": sections[t], "stage_m": st, "all_sources": float(np.nanquantile(mv, .95) - np.nanquantile(mv, .05))}
        for src in ALL_SOURCES:
            mv1 = section_mean(chain.sample(st, pts, n=400, sources=(src,), seed=2))
            row[src] = float(np.nanquantile(mv1, .95) - np.nanquantile(mv1, .05))
        att.append(row)
att = pd.DataFrame(att)
att.to_csv(WORK / "uncertainty_attribution.csv", index=False)

# ---------------------------------------------------------------- 4. numbers quoted in the paper
width = ser.mean_v_hi - ser.mean_v_lo
inside = (ser.stage_m >= SURVEYED[0]) & (ser.stage_m <= SURVEYED[1])
reach = ser.groupby("time_utc").agg(stage_m=("stage_m", "first"), v=("mean_v_med", "mean"))  # reach-average section velocity
reach.index = pd.to_datetime(reach.index, utc=True)
monthly = reach.v.groupby(reach.index.strftime("%Y-%m")).median()
sec_in = ser[inside].groupby("transect").mean_v_med.median()
smd = df.groupby("transect").section_mean_depth_m.first()
rho = spearmanr(sec_in.values, smd.reindex(sec_in.index).values)
out = {"band_width_90_median_m_s": {"inside_surveyed_range": float(width[inside].median()), "above": float(width[ser.stage_m > SURVEYED[1]].median()),
                                    "below": float(width[ser.stage_m < SURVEYED[0]].median())},
       "heldout_sections": {"inside_90_band": int(covs.in_band.sum()), "n": int(len(covs)),
                            "MAE_m_s": float(np.mean(np.abs(covs.mean_v_measured - covs.mean_v_med)))},
       "heldout_readings": {"n": int(len(RH)), "MAE_m_s": float(np.mean(np.abs(RH.velocity_ms - RH.v_pred))),
                            "coverage95": float(np.mean((RH.velocity_ms >= RH.v_lo) & (RH.velocity_ms <= RH.v_hi))),
                            "mean_abs_deviation_of_readings_m_s": float(np.mean(np.abs(RH.velocity_ms - RH.velocity_ms.median())))},
       "reach_average_velocity_monthly_median_m_s": {k: round(float(v), 3) for k, v in monthly.items()},
       "reach_average_velocity_max_m_s": {"value": float(reach.v.max()), "time": str(reach.v.idxmax())},
       "section_velocity_inside_range_median_m_s": {sections[t]: round(float(v), 3) for t, v in sec_in.sort_values().items()},
       "spearman_section_velocity_vs_section_mean_depth": {"rho": float(rho[0]), "p": float(rho[1])},
       "discrepancy_sd_log": chain.disc_split, "calibration": chain.calib}
json.dump(out, open(WORK / "step4.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["band_width_90_median_m_s", "heldout_sections", "heldout_readings"]}, indent=1))
print(att.round(3).to_string(index=False))
