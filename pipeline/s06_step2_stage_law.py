"""Step 2. Remove the stage.

Each amplitude a mixes the effect of place with the flow on its survey day. A stage law separates the two:
    a = A (s - h0)^gamma
where s is the sensor stage at the survey, h0 a cease-to-flow stage, gamma an exponent and A a rating coefficient that
depends only on location. (h0, gamma) are hyperparameters of the full model, chosen by cross-validation with whole survey
days held out: for each candidate law the GP (step 3) is fitted without one day, its prediction of ln A at that day's
locations is converted to ln a at that day's stage, and scored against the measured ln a with the CRPS. Scoring ln a
keeps candidate laws comparable. Every law within 0.01 CRPS of the best, with h0 below the lowest stage on record, is
retained and carried into the Monte Carlo (step 4).

Independent check: the ADCP's 2 h velocity record at one fixed point, where only the stage varies, fitted with the same
form, ln v = c + gamma ln(s - h0).

Inputs:  outputs/intermediate/features.csv, locations.csv, stage_adcp_10min.csv
Outputs: outputs/intermediate/stage_law_grid.csv, stage_law.json (selected and retained laws), adcp_stage_law.csv"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from riverml.config import WORK, H0_GRID, GAMMA_GRID, RIDGE_CRPS, LOWEST_STAGE_M, ADCP_H0_GRID
from riverml.gp import load_points, with_target, Model, scores
warnings.filterwarnings("ignore")

df = load_points()
days = sorted(df.date.unique())


def day_out(h0, gamma, n_restarts):
    """Hold out each survey day; score the predicted ln a (= ln A + ln stage factor) against the measured ln a."""
    d = with_target(df, h0, gamma)
    MU, SD = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for day in days:
        tr, te = np.where(d.date != day)[0], np.where(d.date == day)[0]
        mu, sd = Model(n_restarts=n_restarts).fit(d.iloc[tr]).predict(d.iloc[te])
        MU[te], SD[te] = mu, sd
    return scores(np.log(d.a.values), MU + np.log(d.g.values), np.sqrt(SD ** 2 + d.sd_y.values ** 2))


rows = []
for h0 in H0_GRID:
    for gamma in GAMMA_GRID:
        rows.append({"h0": h0, "gamma": gamma, **day_out(h0, gamma, n_restarts=2)})
        print(f"h0 {h0:.2f}, gamma {gamma:.2f}: day-out CRPS {rows[-1]['CRPS']:.3f}")
grid = pd.DataFrame(rows).sort_values("CRPS").reset_index(drop=True)
grid.to_csv(WORK / "stage_law_grid.csv", index=False)
best = grid.iloc[0]
ridge = grid[(grid.CRPS <= best.CRPS + RIDGE_CRPS) & (grid.h0 < LOWEST_STAGE_M)][["h0", "gamma"]].values.tolist()
no_law = day_out(None, None, n_restarts=3)

# survey-day medians of ln a rise with the stage; after de-staging, ln A should not
pos = df.copy()
pos["A"] = pos.a / np.clip(pos.stage_m - best.h0, 0.02, None) ** best.gamma
dm = pos.groupby("date").agg(stage=("stage_m", "median"), ln_a=("a", lambda v: np.log(v).median()),
                             ln_A=("A", lambda v: np.log(v).median()))

# ADCP check: the same law fitted to the ADCP's 2 h means at its fixed location
g = pd.read_csv(WORK / "stage_adcp_10min.csv", parse_dates=["time_utc"]).set_index("time_utc")
h2 = g.dropna(subset=["adcp_ms", "stage_m"])[["stage_m", "adcp_ms"]].resample("2h").mean().dropna()
adcp = []
for h0 in np.round(np.arange(ADCP_H0_GRID[0], ADCP_H0_GRID[1] + ADCP_H0_GRID[2] / 2, ADCP_H0_GRID[2]), 2):
    x, y = np.log(h2.stage_m - h0), np.log(h2.adcp_ms)
    gam, c = np.polyfit(x, y, 1)
    adcp.append({"h0": h0, "gamma": gam, "rmse_log": float(np.sqrt(np.mean((y - gam * x - c) ** 2)))})
adcp = pd.DataFrame(adcp)
adcp.to_csv(WORK / "adcp_stage_law.csv", index=False)
near_best = adcp[adcp.rmse_log <= adcp.rmse_log.min() * 1.02]          # laws within 2% of the best fit

out = {"selected": {"h0": float(best.h0), "gamma": float(best.gamma), "day_out_CRPS": float(best.CRPS)},
       "retained": ridge, "no_law_day_out": no_law,
       "spearman_day_median_vs_stage": {"ln_a_before": float(spearmanr(dm.stage, dm.ln_a)[0]),
                                        "ln_A_after": float(spearmanr(dm.stage, dm.ln_A)[0]), "days": int(len(dm))},
       "adcp_check": {"n_2h_means": int(len(h2)), "best_h0_range": [float(near_best.h0.min()), float(near_best.h0.max())],
                      "gamma_range_at_best": [float(near_best.gamma.min()), float(near_best.gamma.max())],
                      "gamma_at_selected_h0": float(adcp.loc[np.isclose(adcp.h0, best.h0), "gamma"].iloc[0])}}
json.dump(out, open(WORK / "stage_law.json", "w"), indent=1)
print(json.dumps(out, indent=1))
