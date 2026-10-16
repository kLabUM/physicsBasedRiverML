"""Step 3 machinery: the Gaussian process on the log rating coefficient, the stage law that defines its target,
and the scores used to evaluate it.

Model for one location x with log-law amplitude a(x) observed at stage s:
    a(x) = A(x) (s - h0)^gamma                                   (the stage law, step 2)
    y(x) = ln A(x) = m(x) + f(x) + noise
    m(x) = weighted least squares on [1, substrate and wake indicators]
    f    ~ GP(0, sigma_f^2 Matern_3/2(x, x'; one length scale per feature) + sigma_n^2 delta)
with a fixed per-point noise equal to the uncertainty of each location's log-law fit."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

from .config import WORK, FEATURES, NOTE_COLS, SUBSTRATES


def load_points():
    """One row per surveyed location with a > 0: GP features, notes, log-law amplitude and its uncertainty."""
    fe = pd.read_csv(WORK / "features.csv")
    lc = pd.read_csv(WORK / "locations.csv")[["location_id", "a", "water_depth_m", "stage_m", "x_m", "y_m", "across_m",
                                               "bed_rtk_sensor_datum_m", "bed_elev_m", "point"]]
    df = fe.drop(columns=["across_m"]).merge(lc, on="location_id")
    df = df[df.a > 0].reset_index(drop=True)
    for s in SUBSTRATES:
        df[f"sub_{s}"] = (df.substrate == s).astype(float)
    return df


def stage_factor(stage, h0, gamma):
    """(s - h0)^gamma, floored at 0.02 m so the factor stays positive near the cease-to-flow stage."""
    return np.clip(np.asarray(stage, float) - h0, 0.02, None) ** gamma


def with_target(df, h0=None, gamma=None):
    """Add the GP target y = ln A and its sd. h0=None means no stage law (A = a)."""
    d = df.copy()
    d["g"] = np.ones(len(d)) if h0 is None else stage_factor(d.stage_m.values, h0, gamma)
    d["A"] = d.a / d.g
    d["y"] = np.log(d.A)
    d["sd_y"] = np.clip(d.a_sd / d.a, 0.02, 1.0)        # relative error of a = sd of ln a = sd of ln A
    return d


def _wls(X, y, w):
    W = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)
    return b


class Model:
    """Weighted least-squares mean on the field notes plus an ARD Matern 3/2 GP on the geometry features.
    feats=None gives the mean alone (the reach-mean baseline when notes=False)."""

    def __init__(self, feats=FEATURES, notes=True, n_restarts=3, nu=1.5):
        self.feats, self.notes, self.n_restarts, self.nu = feats, notes, n_restarts, nu

    def design(self, d):
        return np.column_stack([np.ones(len(d))] + ([d[c].values for c in NOTE_COLS] if self.notes else []))

    def fit(self, d):
        w = 1 / d.sd_y.values ** 2
        self.b = _wls(self.design(d), d.y.values, w)
        r = d.y.values - self.design(d) @ self.b
        if self.feats is None:
            self.rsd = float(np.sqrt(np.average(r ** 2, weights=w)))
            return self
        X = d[self.feats].values.astype(float)
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-9         # features standardised
        self.ys = r.std() + 1e-9                              # residuals scaled to unit sd
        k = (C(1.0, (1e-2, 1e2)) * Matern(np.ones(len(self.feats)), [(0.05, 50.0)] * len(self.feats), nu=self.nu)
             + WhiteKernel(0.1, (1e-4, 4.0)))                 # amplitude x Matern(ARD) + nugget
        self.gp = GaussianProcessRegressor(k, alpha=(d.sd_y.values / self.ys) ** 2,    # fixed per-point noise
                                           n_restarts_optimizer=self.n_restarts, random_state=0)
        self.gp.fit((X - self.mx) / self.sx, r / self.ys)
        return self

    def predict(self, d):
        mu = self.design(d) @ self.b
        if self.feats is None:
            return mu, np.full(len(d), self.rsd)
        m, sd = self.gp.predict((d[self.feats].values.astype(float) - self.mx) / self.sx, return_std=True)
        return mu + m * self.ys, sd * self.ys

    def predict_cov(self, d):
        mu = self.design(d) @ self.b
        m, cov = self.gp.predict((d[self.feats].values.astype(float) - self.mx) / self.sx, return_cov=True)
        return mu + m * self.ys, cov * self.ys ** 2

    def hyperparameters(self):
        """Length scales in feature units, signal and nugget sd in ln A, and the log marginal likelihood."""
        k = self.gp.kernel_
        ls = np.atleast_1d(k.k1.k2.length_scale) * self.sx
        return {"signal_sd": float(np.sqrt(k.k1.k1.constant_value) * self.ys),
                "nugget_sd": float(np.sqrt(k.k2.noise_level) * self.ys),
                "log_marginal_likelihood": float(self.gp.log_marginal_likelihood_value_),
                **{f: float(l) for f, l in zip(self.feats, ls)}}


def crps_normal(y, mu, sd):
    """Continuous ranked probability score of a normal forecast (lower is better)."""
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def scores(y, mu, sd):
    z = (y - mu) / sd
    return {"CRPS": float(np.mean(crps_normal(y, mu, sd))), "MAE": float(np.mean(np.abs(y - mu))),
            "coverage95": float(np.mean(np.abs(z) <= 1.96)), "mean_sd": float(np.mean(sd)), "n": int(len(y))}
