"""Run the whole analysis, from the raw data to the paper's figures and numbers (about 10 minutes on a laptop).

    python run_all.py             # everything
    python run_all.py --figures   # figures only, from existing outputs/intermediate/
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = [
    "s01_stage_record.py",            # data: water level record cleaned; ADCP on the same grid
    "s02_field_survey.py",            # data: velocity readings and nests, with the stage at each reading
    "s03_channel_geometry.py",        # data: main-stem centerline, bank lines, channel polygon
    "s04_step1_log_law.py",           # step 1: one roughness length; one amplitude a per location; bed on the stage datum
    "s05_geometry_features.py",       # inputs for step 3: geometry features at each location
    "s06_step2_stage_law.py",         # step 2: the stage law a = A (s - h0)^gamma, chosen by day-out cross-validation
    "s07_step3_gaussian_process.py",  # step 3: GP on ln A, section/random/day cross-validation, map of A
    "s08_step4_monte_carlo.py",       # step 4: section velocities every 2 h with bands; held-out-day validation
    "s09_step5_nest_suitability.py",  # step 5: nest suitability learned from nests; E through time and against stage
    "s10_paper_numbers.py",           # every number quoted in the paper, in outputs/results/
]
FIGURES = ["fig1_method.py", "fig2_study_area.py", "fig3_steps1_2.py", "fig4_step3.py", "fig5_step4.py",
           "fig6_step5.py", "figS_log_law_profiles.py"]

scripts = ([] if "--figures" in sys.argv else [ROOT / "pipeline" / s for s in PIPELINE]) + [ROOT / "figures" / s for s in FIGURES]
t0 = time.time()
for s in scripts:
    print(f"\n=== {s.parent.name}/{s.name}  [{time.time() - t0:.0f} s]", flush=True)
    subprocess.run([sys.executable, str(s)], check=True)
print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Figures in outputs/figures/, numbers in outputs/results/.")
