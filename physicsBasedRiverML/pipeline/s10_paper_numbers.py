"""Collect every number quoted in the paper from the pipeline's outputs, so each can be traced to the step that made it.

Inputs:  outputs/intermediate/*.json, cv_scores.csv, cv_sections.csv, uncertainty_attribution.csv
Outputs: outputs/results/table2_cross_validation.csv, outputs/results/paper_numbers.md"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from riverml.config import WORK, RESULTS

J = lambda f: json.load(open(WORK / f))
ll, ch, law, s4, s5 = J("log_law.json"), J("channel_summary.json"), J("stage_law.json"), J("step4.json"), J("step5.json")
cv = pd.read_csv(WORK / "cv_scores.csv").set_index(["scheme", "model"])
sec = pd.read_csv(WORK / "cv_sections.csv")
att = pd.read_csv(WORK / "uncertainty_attribution.csv")
sm = {m: g for m, g in sec.groupby("model")}
cvv = lambda s, m, c: float(cv.loc[(s, m), c])
n_sec, n_days = sec.transect.nunique(), int(law["spearman_day_median_vs_stage"]["days"])

# ---------------------------------------------------------------- Table 2
t2 = pd.DataFrame([
    [f"Space: one section held out ({n_sec} folds)", "Points", "CRPS of ln A", cvv("section_out", "six features", "CRPS"), f"{cvv('section_out', 'reach mean', 'CRPS'):.3f} (reach mean)"],
    ["", "Points", "95% interval coverage", cvv("section_out", "six features", "coverage95"), f"{cvv('section_out', 'reach mean', 'coverage95'):.2f} (reach mean)"],
    ["", f"Sections (n = {n_sec})", "Mean in 90% interval", f"{int(sm['six features'].in90.sum())} / {n_sec}", f"{int(sm['bank distance'].in90.sum())} / {n_sec} (bank distance)"],
    ["", "Sections", "MAE of section mean ln A", float(np.mean(np.abs(sm['six features'].obs - sm['six features'].pred))),
     f"{np.mean(np.abs(sm['bank distance'].obs - sm['bank distance'].pred)):.2f} (bank distance)"],
    ["", "Points", "CRPS, random 10-fold", cvv("random_10fold", "six features", "CRPS"), f"{cvv('random_10fold', 'reach mean', 'CRPS'):.3f} (reach mean)"],
    [f"Time: one survey day held out ({n_days} folds)", "Points", "CRPS of ln a", cvv("day_out", "six features", "CRPS"), f"{law['no_law_day_out']['CRPS']:.3f} (no stage law)"],
    ["", "Points", "95% interval coverage", cvv("day_out", "six features", "coverage95"), f"{law['no_law_day_out']['coverage95']:.2f} (no stage law)"],
    ["", f"Readings (n = {s4['heldout_readings']['n']:,})", "MAE of velocity (m/s)", s4["heldout_readings"]["MAE_m_s"],
     f"{s4['heldout_readings']['mean_abs_deviation_of_readings_m_s']:.3f} (spread of readings)"],
    ["", "Readings", "95% interval coverage", s4["heldout_readings"]["coverage95"], ""],
    ["", f"Sections (n = {n_sec})", "Mean velocity in 90% interval", f"{s4['heldout_sections']['inside_90_band']} / {s4['heldout_sections']['n']}", ""],
    ["", "Sections", "MAE of mean velocity (m/s)", s4["heldout_sections"]["MAE_m_s"], ""]],
    columns=["scale", "unit", "metric", "six-feature model", "baseline"])
t2.to_csv(RESULTS / "table2_cross_validation.csv", index=False)

# ---------------------------------------------------------------- every other number, by paper section
f2, f3 = lambda x: f"{x:.2f}", lambda x: f"{x:.3f}"
pct = lambda x: f"{x * 100:.0f}%"
inrange = att[att.stage_m <= 1.10]
rows = [
    ("Data", "readings / not used / used", f"{ll['readings']} / {ll['readings_not_used']} / {ll['readings_used']}"),
    ("Data", "locations surveyed / with a fitted amplitude / still water", f"{ll['locations']} / {ll['locations_fitted']} / {ll['locations_still']}"),
    ("Data", "fitted locations 2025 / 2026", " / ".join(str(v) for v in ll["locations_by_year_fitted"].values())),
    ("Data", "locations with 3+ readings / a single reading", f"{ll['locations_3plus_readings']} / {ll['locations_single_reading']}"),
    ("Data", "sections / survey days", f"{ll['sections']} / {ll['survey_days']}"),
    ("Data", "nests in wetted cells / available cells per nest", f"{s5['nests_used']} / {s5['available_cells'] // s5['nests_used']}"),
    ("Data", "bank points placed / channel median width / area", f"{ch['bank_points_placed']} / {ch['median_width_m']} m / {ch['area_ha']} ha"),
    ("Step 1", "roughness length z0 (within 5% of best)", f"{ll['z0_m'] * 1000:.1f} mm ({ll['z0_within_5pct_m'][0] * 1000:.1f}-{ll['z0_within_5pct_m'][1] * 1000:.1f} mm)"),
    ("Step 1", "R2 quartiles, profiles with 3+ readings", " / ".join(f2(x) for x in ll["fit_r2_quartiles"])),
    ("Step 1", "fits with R2 < 0; median RMSE of those / others (m/s)", f"{ll['fits_r2_below_0']}; {f3(ll['fit_rmse_median_r2_below_0'])} / {f3(ll['fit_rmse_median_others'])}"),
    ("Step 1", "pooled 1 ft transfer error (m/s), profiles", f"{f3(ll['transfer_check']['MAE_m_s'])}, {ll['transfer_check']['n_profiles']}"),
    ("Step 1", "amplitude a median (IQR), m/s", f"{f3(ll['amplitude_a_median_iqr'][0])} ({f3(ll['amplitude_a_median_iqr'][1])}-{f3(ll['amplitude_a_median_iqr'][2])})"),
    ("Step 2", "selected law h0, gamma; day-out CRPS", f"{law['selected']['h0']}, {law['selected']['gamma']}; {f3(law['selected']['day_out_CRPS'])}"),
    ("Step 2", "day-out CRPS / coverage without a law", f"{f3(law['no_law_day_out']['CRPS'])} / {f2(law['no_law_day_out']['coverage95'])}"),
    ("Step 2", "retained laws (h0, gamma)", "; ".join(f"({a}, {b})" for a, b in law["retained"])),
    ("Step 2", "Spearman rho of day medians vs stage: ln a / ln A", f"{f2(law['spearman_day_median_vs_stage']['ln_a_before'])} / {f2(law['spearman_day_median_vs_stage']['ln_A_after'])}"),
    ("Step 2", "ADCP check: best h0 range, gamma range; gamma at selected h0",
     f"{law['adcp_check']['best_h0_range'][0]}-{law['adcp_check']['best_h0_range'][1]} m, {f2(law['adcp_check']['gamma_range_at_best'][0])}-{f2(law['adcp_check']['gamma_range_at_best'][1])}; {f2(law['adcp_check']['gamma_at_selected_h0'])}"),
    ("Step 3", "point CRPS, bank distance alone (section held out)", f3(cvv("section_out", "bank distance", "CRPS"))),
    ("Step 4", "90% band width, section mean velocity: inside / above / below surveyed range (m/s)",
     " / ".join(f2(s4["band_width_90_median_m_s"][k]) for k in ["inside_surveyed_range", "above", "below"])),
    ("Step 4", "reach-average section velocity, monthly median: Apr 2026 / Jun 2025 / Aug 2026 (m/s)",
     " / ".join(f2(s4["reach_average_velocity_monthly_median_m_s"][k]) for k in ["2026-04", "2025-06", "2026-08"])),
    ("Step 4", "slowest three / fastest section inside the surveyed range (m/s)",
     ", ".join(f"{k} {v:.2f}" for k, v in list(s4["section_velocity_inside_range_median_m_s"].items())[:3]) + "; " +
     ", ".join(f"{k} {v:.2f}" for k, v in list(s4["section_velocity_inside_range_median_m_s"].items())[-1:])),
    ("Step 4", "Spearman rho, section velocity vs section mean depth (p)",
     f"{f2(s4['spearman_section_velocity_vs_section_mean_depth']['rho'])} ({f2(s4['spearman_section_velocity_vs_section_mean_depth']['p'])})"),
    ("Step 4", "largest single sources of the band at stages <= 1.10 m (median width, m/s)",
     ", ".join(f"{s} {inrange[s].median():.3f}" for s in inrange[["A_field", "disc", "bed", "stage", "law", "z0"]].median().sort_values(ascending=False).index)),
    ("Step 4", "law spread at 1.30 m vs discrepancy at 1.30 m (median width, m/s)", f"{att[att.stage_m == 1.30].law.median():.3f} vs {att[att.stage_m == 1.30].disc.median():.3f}"),
    ("Step 5", "velocity coefficient b_v (per m/s)", f"{s5['velocity_coef_per_m_s'][0]:.1f} +/- {s5['velocity_coef_per_m_s'][1]:.1f}"),
    ("Step 5", "suitability halves at (90% interval), m/s", f"{f2(s5['suitability_halves_at_m_s']['median'])} ({f2(s5['suitability_halves_at_m_s']['ci90'][0])}-{f2(s5['suitability_halves_at_m_s']['ci90'][1])})"),
    ("Step 5", "E at 0.70 / 0.90 (90% band) / 1.10 m",
     f"{pct(s5['E_at_stage']['0.70'][0])} / {pct(s5['E_at_stage']['0.90'][0])} ({pct(s5['E_at_stage']['0.90'][1])}-{pct(s5['E_at_stage']['0.90'][2])}) / {pct(s5['E_at_stage']['1.10'][0])}"),
    ("Step 5", "E season median 2025 / 2026", f"{pct(s5['E_season_median']['2025'])} / {pct(s5['E_season_median']['2026'])}"),
    ("Step 5", "E monthly median Apr 2026 / Aug 2026", f"{pct(s5['E_monthly_median']['2026-04'])} / {pct(s5['E_monthly_median']['2026-08'])}"),
    ("Step 5", "nest preference with survey days as the unit: t-test p", f2(s5["nest_rank_by_survey_day"]["t_test_p_vs_0.5"])),
]
md = ["# Numbers quoted in the paper", "", "Regenerated by `pipeline/s10_paper_numbers.py`. Table 1 and Table 2 are in this folder as CSV.", "",
      "| section | quantity | value |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
md += ["", "## Table 2. Cross-validation through space and time", "", "| " + " | ".join(t2.columns) + " |", "|" + "---|" * len(t2.columns)]
md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in t2.itertuples(index=False)]
open(RESULTS / "paper_numbers.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
