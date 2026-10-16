"""Paths and constants shared by every script. Change a study setting here, nowhere else."""
from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"                                   # the only external inputs
OUT = ROOT / "outputs"
WORK = OUT / "intermediate"                           # tables written by the pipeline and read by later steps
FIGURES = OUT / "figures"                             # the paper's six figures and the supplementary pages
RESULTS = OUT / "results"                             # the numbers quoted in the paper (tables and a summary)
for _p in (WORK, FIGURES, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

SURVEY_FILES = {2025: RAW / "25_ecohydroData.csv", 2026: RAW / "26_ecohydroData.csv"}
STAGE_FILES = [RAW / "25_depthSensorData_Middle.csv", RAW / "26_depthSensorData_Middle.csv"]
ADCP_FILE = RAW / "25_ADCPData.csv"
RIVER_NETWORK = RAW / "Huron_RiverSystem.geojson"     # NHD flowlines of the Huron River
BASEMAP = RAW / "basemap"                             # Figure 2a only: an ArcGIS export and its georeference

# ------------------------------------------------------------------ units, frames, site
FT = 0.3048                                           # metres per foot (survey feet differ by 2 ppm: negligible here)
CRS = "ESRI:102690"                                   # NAD83 StatePlane Michigan South, feet: the survey's horizontal frame
TZ_SURVEY = "America/Detroit"                         # survey sheets record local time
ADCP_E, ADCP_N = 13244857.0, 319995.5                 # ADCP position (StatePlane ft); origin of the local x, y frame
SENSOR_LON, SENSOR_LAT = -83.9112202, 42.3871931     # bridge-mounted stage sensor (network site ARB012), ~2 km upstream
SITE_LAT, SITE_LON = 42.38, -83.92                    # for the sun's elevation (night detection)
NORTHING_BAND = (280000, 323000)                      # survey rows in this band (StatePlane ft) belong to the study reach

# ------------------------------------------------------------------ stage record
STAGE_DATUM_MM = 7313.2                               # sensor mounting height above the stage datum, mm
STAGE_PLAUSIBLE_M = (0.3, 3.0)                        # outside this a reading is a sensor fault
SPIKE_M = 0.15                                        # departure from the 6 h rolling median that marks a spike
STAGE_SD = 0.015                                      # stage uncertainty carried into the Monte Carlo (m)

# ------------------------------------------------------------------ step 1: log law
Z_MIN, V_MIN = 0.03, 0.005                            # readings below 0.03 m or at/below 0.005 m/s are not used
Z_REF = 0.3048                                        # 1 ft above the bed: the height of the pooled transfer check
Z0_GRID = (1e-4, 1e-1, 61)                            # log-spaced grid searched for the roughness length (m)

# ------------------------------------------------------------------ step 2: stage law a = A (s - h0)^gamma
H0_GRID = [0.35, 0.45, 0.55, 0.65]
GAMMA_GRID = [0.5, 0.75, 1.0, 1.25]
RIDGE_CRPS = 0.01                                     # laws within this day-out CRPS of the best are all retained
LOWEST_STAGE_M = 0.66                                 # retained laws need h0 below the lowest stage the river reached
ADCP_H0_GRID = (0.35, 0.65, 0.05)                     # h0 values tried when fitting the stage law to the ADCP record

# ------------------------------------------------------------------ step 3: Gaussian process
STAGE_REF = 0.90                                      # depth features are computed at this common stage (m)
FEATURES = ["along_m", "rel_across", "dist_bank_m", "rel_depth", "section_mean_depth_m", "lateral_bed_slope_abs"]
SUBSTRATES = ["cobble", "gravel", "sand", "silt", "vegetation", "woody debris"]
NOTE_COLS = [f"sub_{s}" for s in SUBSTRATES] + ["wake_flag"]
BED_SD = 0.06                                         # RTK bed elevation uncertainty (m)

# ------------------------------------------------------------------ step 4: Monte Carlo
Z0_RANGE = (0.007, 0.016)                             # roughness lengths within 5% of the best fit (step 1: 7.1-15.8 mm), rounded; sampled log-uniformly
Z_MEAS = 0.25                                         # typical height of the readings that set a (m); rescales a when z0 is redrawn
SURVEYED = (0.74, 1.09)                               # stage range covered by the surveys (m)
N_DRAWS = 100                                         # Monte Carlo draws for the time series

# ------------------------------------------------------------------ step 5: nests
N_AVAILABLE = 20                                      # available cells drawn for each nest (used-available design)
SEED = 0
