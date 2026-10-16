"""Step 5. Transform downstream: velocity into a decision variable, here smallmouth bass nesting suitability.

Used-available design: each nest (used, 1) is compared with 20 wetted grid cells drawn at random at the nest's own stage
(available, 0), in a logistic model on depth-averaged velocity, bank distance and depth with weak, zero-centred priors:
    log p(b | y) = sum_i [y_i z_i - ln(1 + exp z_i)] - sum_j (b_j - mu_j)^2 / (2 sigma_j^2)
and a Laplace (normal) approximation to the posterior. Holding bank distance and depth fixed, the relative suitability
of a cell at velocity v with respect to still water is S(v) = exp(b_v v). The decision variable E is the wetted-width-
and section-length-weighted mean of S over the reach, carried through the same Monte Carlo draws as step 4 with one b_v
drawn from its posterior per draw: E against stage (the ecological rating curve) and E every 2 h through both seasons.

A check on the nests' clustering by survey day: each nest's predicted velocity is ranked among all wetted cells at its
stage; the day means of those ranks are tested against 0.5 (no preference).

Inputs:  outputs/intermediate/chain.pkl, grid.csv, nests.csv, stage_adcp_10min.csv
Outputs: outputs/intermediate/nests_used.csv, suitability_curve.csv, suitability_2h.csv, step5.json"""
import sys, json, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import ttest_1samp
from riverml.config import WORK, N_AVAILABLE, SEED, STAGE_SD, BED_SD, Z0_RANGE, Z_MEAS
from riverml.chain import width_weights, length_weights
from riverml.gp import stage_factor
warnings.filterwarnings("ignore")

chain = pickle.load(open(WORK / "chain.pkl", "rb"))
df, Z0 = chain.df, chain.z0
H0, GAMMA = chain.primary
grid = pd.read_csv(WORK / "grid.csv")
g10 = pd.read_csv(WORK / "stage_adcp_10min.csv", parse_dates=["time_utc"]).set_index("time_utc")
g10.index = pd.to_datetime(g10.index, utc=True)
st10 = g10.stage_m


def v_bar(logA, bed, stage):
    """Depth-averaged velocity under the selected law (NaN where dry), and the depth."""
    h = np.clip(stage - np.asarray(bed), 0, None)
    a = np.exp(logA) * stage_factor(stage, H0, GAMMA)
    return np.where(h > np.e * Z0, a * (np.log(np.clip(h, np.e * Z0, None) / Z0) - 1), np.nan), h


# ---------------------------------------------------------------- 1. each nest, its stage and its grid cell
nests = pd.read_csv(WORK / "nests.csv").drop_duplicates("location_id").copy()


# Many nest records carry only a date, so each nest takes the nearest value of the 10 min stage record within 6 h of its
# time; failing that, the median stage of its day; the one nest day before the sensor was installed (27 May 2025) takes
# the first full day of record (30 May 2025), which had the same base flow at the downstream gauge.
def stage_for(t):
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    i = min(max(st10.index.searchsorted(t), 0), len(st10) - 1)
    return float(st10.iloc[i]) if abs((st10.index[i] - t).total_seconds()) < 6 * 3600 else np.nan


nests["stage_use"] = [stage_for(t) if pd.notna(t) else np.nan for t in nests.time_utc]
day_med = st10.groupby(st10.index.date).median()
day_med.index = pd.to_datetime(day_med.index).strftime("%Y-%m-%d")
nests.loc[nests.stage_use.isna(), "stage_use"] = nests.loc[nests.stage_use.isna(), "date"].map(day_med)   # no time recorded
nests.loc[nests.stage_use.isna() & (nests.date == "2025-05-27"), "stage_use"] = float(st10["2025-05-30"].median())  # before the sensor started
tree = cKDTree(grid[["x_m", "y_m"]].values)
dist, _ = tree.query(nests[["x_m", "y_m"]].values)
nests = nests[dist < 6].copy()                                       # nests on the mapped channel
cells = grid.iloc[tree.query(nests[["x_m", "y_m"]].values)[1]].reset_index(drop=True)
nests = nests.reset_index(drop=True)
nests["v_pred_ms"] = [float(v_bar(c.logA, c.bed_m, s)[0]) for s, (_, c) in zip(nests.stage_use, cells.iterrows())]
pct = []                                                             # rank of the nest's velocity among wetted cells
for s, (_, c), v in zip(nests.stage_use, cells.iterrows(), nests.v_pred_ms):
    vg, _ = v_bar(grid.logA.values, grid.bed_m.values, s)
    pct.append(float(np.mean(vg[~np.isnan(vg)] < v)) if not np.isnan(v) else np.nan)
nests["percentile_in_reach"] = pct
day_means = nests.dropna(subset=["percentile_in_reach"]).groupby("date").percentile_in_reach.mean()
keep = nests.v_pred_ms.notna().values                                # nests in cells wet at their stage
nests, cells = nests[keep].reset_index(drop=True), cells[keep].reset_index(drop=True)
nests.to_csv(WORK / "nests_used.csv", index=False)
print(f"{len(nests)} nests in wetted cells")

# ---------------------------------------------------------------- 2. used-available logistic model, Laplace posterior
rng = np.random.default_rng(SEED)
Xu, Xa = [], []
for i, r in nests.iterrows():
    vg, hg = v_bar(grid.logA.values, grid.bed_m.values, r.stage_use)
    ok = ~np.isnan(vg)
    vc, hc = v_bar(cells.logA.iloc[i], cells.bed_m.iloc[i], r.stage_use)
    Xu.append([1.0, float(vc), cells.dist_bank_m.iloc[i], float(hc)])
    for j in rng.choice(np.where(ok)[0], N_AVAILABLE, replace=False):
        Xa.append([1.0, vg[j], grid.dist_bank_m.values[j], hg[j]])
X = np.array(Xu + Xa)
y = np.r_[np.ones(len(Xu)), np.zeros(len(Xa))]
prior_mean = np.zeros(4)                                             # intercept, velocity (per m/s), bank distance (per m), depth (per m)
prior_sd = np.array([10.0, 10.0, 0.5, 10.0])


def neg_log_post(b):
    z = X @ b
    return float(np.sum(np.logaddexp(0, z) - y * z) + np.sum((b - prior_mean) ** 2 / (2 * prior_sd ** 2)))


def grad(b):
    p = 1 / (1 + np.exp(-(X @ b)))
    return X.T @ (p - y) + (b - prior_mean) / prior_sd ** 2


res = minimize(neg_log_post, np.zeros(4), jac=grad, method="BFGS")
p = 1 / (1 + np.exp(-(X @ res.x)))
b_post = res.x
C_post = linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + np.diag(1 / prior_sd ** 2))
bv = rng.multivariate_normal(b_post, C_post, 4000)[:, 1]
v50 = np.log(0.5) / bv[bv < 0]                                       # velocity at which suitability halves
print(f"b_v = {b_post[1]:.1f} +/- {np.sqrt(C_post[1, 1]):.1f} per m/s; suitability halves at {np.median(v50):.2f} m/s")

# ---------------------------------------------------------------- 3. E through the Monte Carlo draws
secs = {t: g.assign(_x=g.x_m.round(2), _y=g.y_m.round(2)).drop_duplicates(["_x", "_y"]).sort_values("across_m").reset_index(drop=True)
        for t, g in df.groupby("transect")}
names = list(secs)
pts = pd.concat([secs[t].assign(_sec=t) for t in names], ignore_index=True)
npts = len(pts)
lw = dict(zip(names, length_weights(secs)))
weight = np.concatenate([width_weights(secs[t]) for t in names]) * np.array([lw[s] for s in pts._sec])   # width x section length


def suitability(stages, n, seed, chunk=150):
    """(n, T) draws of E: the same joint draws as step 4 (law, GP field, discrepancy, bed, stage, z0) plus one b_v per draw."""
    rg = np.random.default_rng(seed)
    keys = [chain.ridge[i] for i in rg.integers(len(chain.ridge), size=n)]
    mus, Ls = {}, {}
    for key in set(keys):
        mu, cov = chain.models[key].predict_cov(pts)
        mus[key], Ls[key] = mu, linalg.cholesky(cov + 1e-8 * np.eye(npts), lower=True)
    logA = (np.stack([mus[k] + Ls[k] @ rg.standard_normal(npts) for k in keys])
            + rg.normal(0, chain.disc_split["independent_sd_log"], (n, npts)) + rg.normal(0, chain.disc_split["common_sd_log"], (n, 1)))
    bed = pts.bed_m.values[None, :] + rg.normal(0, BED_SD, (n, npts))
    z0 = np.exp(rg.uniform(np.log(Z0_RANGE[0]), np.log(Z0_RANGE[1]), n))
    A = np.exp(logA) * (np.log(Z_MEAS / Z0) / np.log(Z_MEAS / z0))[:, None]
    h0, gam = np.array([k[0] for k in keys]), np.array([k[1] for k in keys])
    bvd = rg.multivariate_normal(b_post, C_post, n)[:, 1]
    E = np.full((n, len(stages)), np.nan)
    stages = np.asarray(stages, float)
    for j0 in range(0, len(stages), chunk):
        js = np.arange(j0, min(j0 + chunk, len(stages)))
        st = stages[js][None, :] + rg.normal(0, STAGE_SD, (n, 1))
        g = stage_factor(st, h0[:, None], gam[:, None])
        h = np.clip(st[:, :, None] - bed[:, None, :], 0, None)
        wet = h > np.e * z0[:, None, None]
        v = np.where(wet, A[:, None, :] * g[:, :, None] * (np.log(np.clip(h, 1e-6, None) / z0[:, None, None]) - 1), 0.0)
        w = weight[None, None, :] * wet
        E[:, js] = (np.exp(bvd[:, None, None] * v) * w).sum(2) / np.maximum(w.sum(2), 1e-12)
    return E


grid_st = np.linspace(0.65, 1.35, 15)
Eg = suitability(grid_st, 300, 1)
curve = pd.DataFrame({"stage_m": grid_st, "E_med": np.nanmedian(Eg, 0), "E_lo": np.nanquantile(Eg, .05, 0), "E_hi": np.nanquantile(Eg, .95, 0)})
curve.to_csv(WORK / "suitability_curve.csv", index=False)
st2 = st10.resample("2h").mean().dropna()
st2 = st2[st2.index.year >= 2025]
Es = suitability(st2.values, 100, 2)
ser = pd.DataFrame({"time_utc": st2.index, "stage_m": st2.values, "E_med": np.nanmedian(Es, 0), "E_lo": np.nanquantile(Es, .05, 0), "E_hi": np.nanquantile(Es, .95, 0)})
ser.to_csv(WORK / "suitability_2h.csv", index=False)

at = lambda s: curve.iloc[int(np.argmin(np.abs(grid_st - s)))]
monthly = ser.set_index("time_utc").E_med.groupby(lambda t: t.strftime("%Y-%m")).median()
out = {"nests_used": int(len(nests)), "available_cells": int(len(Xa)),
       "velocity_coef_per_m_s": [float(b_post[1]), float(np.sqrt(C_post[1, 1]))],
       "bank_distance_coef_per_m": [float(b_post[2]), float(np.sqrt(C_post[2, 2]))], "depth_coef_per_m": [float(b_post[3]), float(np.sqrt(C_post[3, 3]))],
       "suitability_halves_at_m_s": {"median": float(np.median(v50)), "ci90": [float(np.quantile(v50, .05)), float(np.quantile(v50, .95))]},
       "E_at_stage": {f"{s:.2f}": [float(at(s).E_med), float(at(s).E_lo), float(at(s).E_hi)] for s in (0.70, 0.90, 1.10)},
       "E_season_median": {str(yr): float(np.nanmedian(ser[ser.time_utc.dt.year == yr].E_med)) for yr in (2025, 2026)},
       "E_monthly_median": {k: round(float(v), 3) for k, v in monthly.items()},
       "nest_rank_by_survey_day": {"days": int(len(day_means)), "mean_of_day_means": float(day_means.mean()),
                                   "t_test_p_vs_0.5": float(ttest_1samp(day_means, 0.5).pvalue)}}
json.dump(out, open(WORK / "step5.json", "w"), indent=1)
print(json.dumps(out, indent=1))
