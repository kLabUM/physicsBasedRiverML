"""Step 3. Learn the place.

Gaussian process regression maps the log rating coefficient ln A (step 2) onto the six channel-geometry features
(riverml/gp.py). Evaluated three ways:
  * one river section held out at a time: prediction where no nearby survey exists, the relevant test;
    random 10-fold cross-validation shown for contrast (neighbouring points leak information);
  * section means: the held-out section's mean ln A against its 90% interval from the joint posterior covariance;
  * one survey day held out at a time: prediction of an unsurveyed day from its stage alone.
Baselines: the reach mean, and a GP on bank distance alone.

Then the fitted models for every retained stage law are kept for the Monte Carlo (step 4), and A is mapped on a 2 m
grid inside the channel polygon, with the bed between transects from a second GP on the RTK bed elevations.

Inputs:  outputs/intermediate/features.csv, locations.csv, stage_law.json, log_law.json, centerline.geojson,
         bank_lines.geojson, channel.geojson, rtk_bank_points.csv
Outputs: outputs/intermediate/cv_points.csv, cv_scores.csv, cv_sections.csv, chain.pkl, grid.csv,
         outputs/results/table1_hyperparameters.csv"""
import sys, json, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from riverml.config import WORK, RESULTS, FT, ADCP_E, ADCP_N, STAGE_REF, FEATURES, NOTE_COLS, BED_SD
from riverml.gp import load_points, with_target, Model, crps_normal
from riverml.chain import Chain
from riverml.geometry import reach_coordinates
warnings.filterwarnings("ignore")

law = json.load(open(WORK / "stage_law.json"))
H0, GAMMA = law["selected"]["h0"], law["selected"]["gamma"]
Z0 = json.load(open(WORK / "log_law.json"))["z0_m"]
df = load_points()
d = with_target(df, H0, GAMMA)

# ---------------------------------------------------------------- 1. point-level cross-validation (space)
MODELS = {"reach mean": None, "bank distance": ["dist_bank_m"], "six features": FEATURES}
folds = {"section_out": d.transect.values,
         "random_10fold": np.random.default_rng(0).permutation(np.arange(len(d)) % 10)}
rows, preds = [], []
for scheme, groups in folds.items():
    for name, feats in MODELS.items():
        mu, sd = np.zeros(len(d)), np.zeros(len(d))
        for g in np.unique(groups):
            te = groups == g
            tr = ~te
            if feats is None:                                    # reach mean: weighted mean and sd of the training points
                mu[te] = float(np.average(d.y[tr], weights=1 / d.sd_y[tr] ** 2))
                sd[te] = float(d.y[tr].std())
            else:
                mu[te], sd[te] = Model(feats=feats, n_restarts=1).fit(d[tr]).predict(d[te])
        tot = np.sqrt(sd ** 2 + d.sd_y.values ** 2)              # the held-out value carries its own measurement noise
        rows.append({"scheme": scheme, "model": name, "CRPS": float(crps_normal(d.y.values, mu, tot).mean()),
                     "MAE": float(np.abs(d.y - mu).mean()), "coverage95": float(np.mean(np.abs((d.y - mu) / tot) <= 1.96))})
        preds.append(pd.DataFrame({"scheme": scheme, "model": name, "location_id": d.location_id, "transect": d.transect,
                                   "y": d.y, "mu": mu, "sd": tot}))
cv = pd.DataFrame(rows)

# ---------------------------------------------------------------- 2. day held out (time), six-feature model
mu, sd = np.zeros(len(d)), np.zeros(len(d))
for day in d.date.unique():
    te = (d.date == day).values
    mu[te], sd[te] = Model(n_restarts=1).fit(d[~te]).predict(d[te])
tot = np.sqrt(sd ** 2 + d.sd_y.values ** 2)
cv = pd.concat([cv, pd.DataFrame([{"scheme": "day_out", "model": "six features", "CRPS": float(crps_normal(d.y.values, mu, tot).mean()),
                                   "MAE": float(np.abs(d.y - mu).mean()), "coverage95": float(np.mean(np.abs((d.y - mu) / tot) <= 1.96))}])])
cv.to_csv(WORK / "cv_scores.csv", index=False)
pd.concat(preds).to_csv(WORK / "cv_points.csv", index=False)
print(cv.round(3).to_string(index=False))

# ---------------------------------------------------------------- 3. section means, section held out
rows = []
for name, feats in [("bank distance", ["dist_bank_m"]), ("six features", FEATURES)]:
    for t in sorted(d.transect.unique()):
        te = d[d.transect == t]
        m, cov = Model(feats=feats, n_restarts=1).fit(d[d.transect != t]).predict_cov(te)
        n = len(te)
        var = cov.sum() / n ** 2 + np.sum(te.sd_y ** 2) / n ** 2           # variance of the section mean (joint)
        rows.append({"model": name, "transect": t, "n": n, "obs": te.y.mean(), "pred": m.mean(), "sd": np.sqrt(var)})
S = pd.DataFrame(rows)
S["in90"] = np.abs(S.obs - S.pred) <= 1.645 * S.sd
S.to_csv(WORK / "cv_sections.csv", index=False)
for name, g in S.groupby("model"):
    print(f"section means, {name}: {int(g.in90.sum())}/{len(g)} inside 90%, MAE {np.mean(np.abs(g.obs - g.pred)):.2f}")

# ---------------------------------------------------------------- 4. hyperparameters of the six-feature model (Table 1)
m = Model(n_restarts=3).fit(d)
hp = m.hyperparameters()
rng_ = d[FEATURES].max() - d[FEATURES].min()
t1 = pd.DataFrame([{"feature": f, "range": float(rng_[f]), "length_scale": hp[f], "range_over_length_scale": float(rng_[f] / hp[f])}
                   for f in FEATURES]).sort_values("range_over_length_scale", ascending=False)
t1 = pd.concat([t1, pd.DataFrame([{"feature": "signal sd (ln A)", "length_scale": hp["signal_sd"]},
                                  {"feature": "nugget sd (ln A)", "length_scale": hp["nugget_sd"]},
                                  {"feature": "fixed per-point noise, median (ln A)", "length_scale": float(d.sd_y.median())},
                                  {"feature": "log marginal likelihood", "length_scale": hp["log_marginal_likelihood"]},
                                  {"feature": "stage law h0 (m)", "length_scale": H0}, {"feature": "stage law gamma", "length_scale": GAMMA}])])
t1.rename(columns={"length_scale": "value"}).to_csv(RESULTS / "table1_hyperparameters.csv", index=False)
print(t1.round(3).to_string(index=False))

# ---------------------------------------------------------------- 5. the models for every retained law, for step 4
chain = Chain(df.assign(bed_m=df.bed_rtk_sensor_datum_m.fillna(df.bed_elev_m)), law["retained"], Z0)
pickle.dump(chain, open(WORK / "chain.pkl", "wb"))
print("discrepancy:", json.dumps({k: round(v, 3) for k, v in chain.disc_split.items()}))

# ---------------------------------------------------------------- 6. A on a 2 m grid inside the channel
line = gpd.read_file(WORK / "centerline.geojson").geometry.iloc[0]
bk = gpd.read_file(WORK / "bank_lines.geojson")
left, right = bk[bk.bank == "left"].geometry.iloc[0], bk[bk.bank == "right"].geometry.iloc[0]
poly = gpd.read_file(WORK / "channel.geojson").geometry.iloc[0]
frame = lambda e, n: reach_coordinates(line, e, n, origin=(ADCP_E, ADCP_N))   # m along, m across


# the bed between transects: a GP on the RTK bed elevations and the bank points (along, across)
df["bed_m"] = df.bed_rtk_sensor_datum_m.fillna(df.bed_elev_m)
banks = pd.read_csv(WORK / "rtk_bank_points.csv").dropna(subset=["bank_sensor_datum_m"])
bp = np.column_stack(frame(banks.Easting, banks.Northing))
bed = pd.DataFrame({"along": np.r_[df.along_m.values, bp[:, 0]], "across": np.r_[df.across_m.values, bp[:, 1]],
                    "z": np.r_[df.bed_m.values, banks.bank_sensor_datum_m.values],
                    "sd": np.r_[np.full(len(df), BED_SD), np.full(len(banks), 0.10)]})
Xb = bed[["along", "across"]].values
mxb, sxb, zm, zs = Xb.mean(0), Xb.std(0), bed.z.mean(), bed.z.std()
kb = C(1.0, (1e-2, 1e2)) * Matern([1.0, 1.0], [(0.02, 20.0), (0.02, 20.0)], nu=1.5) + WhiteKernel(0.05, (1e-4, 1.0))
gpb = GaussianProcessRegressor(kb, alpha=(bed.sd.values / zs) ** 2, n_restarts_optimizer=3, random_state=0).fit((Xb - mxb) / sxb, (bed.z.values - zm) / zs)

minx, miny, maxx, maxy = poly.bounds
step = 2.0 / FT
gx, gy = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
gx, gy = gx.ravel(), gy.ravel()
inside = np.array([poly.contains(Point(x, y)) for x, y in zip(gx, gy)])
gx, gy = gx[inside], gy[inside]
P = np.column_stack(frame(gx, gy))
grid = pd.DataFrame({"x_m": (gx - ADCP_E) * FT, "y_m": (gy - ADCP_N) * FT, "along_m": P[:, 0], "across_m": P[:, 1]})
grid = grid[(grid.along_m > df.along_m.min() - 15) & (grid.along_m < df.along_m.max() + 15)].reset_index(drop=True)
pts = [Point(x / FT + ADCP_E, y / FT + ADCP_N) for x, y in zip(grid.x_m, grid.y_m)]
grid["dist_left"] = [left.distance(p) * FT for p in pts]
grid["dist_right"] = [right.distance(p) * FT for p in pts]
grid["dist_bank_m"] = grid[["dist_left", "dist_right"]].min(axis=1)
grid["rel_across"] = (grid.dist_right - grid.dist_left) / (grid.dist_left + grid.dist_right)
mb, sb = gpb.predict((grid[["along_m", "across_m"]].values - mxb) / sxb, return_std=True)
grid["bed_m"], grid["bed_sd_m"] = mb * zs + zm, sb * zs
grid["h_ref_m"] = np.clip(STAGE_REF - grid.bed_m, 0, None)
abin = (grid.along_m // 10) * 10                                     # sections of the grid: 10 m bins along the reach
grid["rel_depth"] = (grid.h_ref_m / grid.groupby(abin).h_ref_m.transform("max").replace(0, np.nan)).fillna(0)
sec = df.groupby("transect").agg(along=("along_m", "median"), smd=("section_mean_depth_m", "first")).sort_values("along")
grid["section_mean_depth_m"] = np.interp(grid.along_m, sec.along, sec.smd)
grid["lateral_bed_slope_abs"] = float(df.lateral_bed_slope_abs.median())
for c in NOTE_COLS:
    grid[c] = float(df[c].mean())                                    # notes are unknown off the transects: the reach-wide share
grid["logA"], grid["logA_sd"] = chain.models[chain.primary].predict(grid)
grid.to_csv(WORK / "grid.csv", index=False)
print(f"grid: {len(grid)} cells of 2 m; median sd of ln A {grid.logA_sd.median():.2f}")
