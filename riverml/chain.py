"""Step 4 machinery: Monte Carlo propagation from the stage record to velocity, with every source of uncertainty.

One draw of the reach:
    ln A   ~ joint GP posterior over all points (one of the retained stage laws, picked at random)
           + model discrepancy (a part common to the whole reach and a part independent at each point,
             calibrated on held-out survey days)
    bed    ~ RTK bed elevation + N(0, 6 cm);  stage ~ record + N(0, 1.5 cm);  z0 ~ log-uniform(7, 16 mm)
    h      = stage - bed                                  local depth
    a      = A (stage - h0)^gamma                         stage law (step 2)
    v_bar  = a [ln(h / z0) - 1]                           depth-averaged velocity (step 1)
In sample_reach, section means are width-weighted over each section's points, and dry points (h < e z0) are excluded."""
import numpy as np
import pandas as pd
from scipy import linalg

from .gp import Model, with_target, stage_factor
from .config import STAGE_SD, BED_SD, Z0_RANGE, Z_MEAS

ALL_SOURCES = ("A_field", "disc", "bed", "stage", "law", "z0")


class Chain:
    """The fitted GPs for every retained stage law, and the calibrated discrepancy.
    df: load_points() with a bed_m column; ridge: retained (h0, gamma) laws, best first; z0: the fitted roughness length."""

    def __init__(self, df, ridge, z0, n_restarts=2):
        self.df, self.ridge, self.z0 = df, [tuple(r) for r in ridge], z0
        self.primary = self.ridge[0]
        self.models = {(h0, g): Model(n_restarts=n_restarts).fit(with_target(df, h0, g)) for h0, g in self.ridge}
        self.calibrate()
        self.disc_split = self.split_discrepancy()

    def _day_out(self, d, day):
        tr, te = np.where(d.date != day)[0], np.where(d.date == day)[0]
        m = Model(n_restarts=1).fit(d.iloc[tr])
        mu, sd = m.predict(d.iloc[te])
        return te, mu, sd

    def calibrate(self):
        """Hold out each survey day under the primary law: how much wider than the GP band are the errors?
        The excess becomes the model-discrepancy sd (in ln A)."""
        d = with_target(self.df, *self.primary)
        Z, SD = [], []
        for day in sorted(d.date.unique()):
            te, mu, sd = self._day_out(d, day)
            st = np.sqrt(sd ** 2 + d.sd_y.values[te] ** 2)
            Z.append((d.y.values[te] - mu) / st)
            SD.append(st)
        Z, SD = np.concatenate(Z), np.concatenate(SD)
        q = float(np.quantile(np.abs(Z), 0.95))
        self.inflate = max(1.0, q / 1.96)
        self.disc_sd = float(np.sqrt(max(self.inflate ** 2 - 1.0, 0.0) * np.mean(SD ** 2)))
        self.calib = {"held_out_coverage95_before": float(np.mean(np.abs(Z) <= 1.96)), "q95_abs_z": q,
                      "inflation": self.inflate, "disc_sd_log": self.disc_sd, "n": int(len(Z))}
        return self.calib

    def split_discrepancy(self):
        """Split the discrepancy into a day-level part common to the reach (the spread of held-out day means,
        less their sampling noise) and a part independent at each point (the rest)."""
        d = with_target(self.df, *self.primary)
        means, sam = [], []
        for day in sorted(d.date.unique()):
            if (d.date == day).sum() < 3:                     # one or two points say nothing about a day level
                continue
            te, mu, _ = self._day_out(d, day)
            r = d.y.values[te] - mu
            means.append(float(np.mean(r)))
            sam.append(float(np.std(r, ddof=1) / np.sqrt(len(r))))
        var_common = max(np.var(means, ddof=1) - np.mean(np.square(sam)), 0.0)
        return {"common_sd_log": float(np.sqrt(var_common)),
                "independent_sd_log": float(np.sqrt(max(self.disc_sd ** 2 - var_common, 0.0))),
                "total_disc_sd_log": self.disc_sd}

    def sample(self, stage, pts, n=500, sources=ALL_SOURCES, seed=0):
        """Draws at ONE stage for one set of points, with each source switchable (used to attribute the band)."""
        rng = np.random.default_rng(seed)
        npts = len(pts)
        out = {"h": np.zeros((n, npts)), "vbar": np.zeros((n, npts))}
        mus, Ls = {}, {}
        for key, m in self.models.items():
            mu, cov = m.predict_cov(pts)
            mus[key], Ls[key] = mu, linalg.cholesky(cov + 1e-8 * np.eye(npts), lower=True)
        bed = pts.bed_m.values
        for k in range(n):
            key = self.ridge[rng.integers(len(self.ridge))] if "law" in sources else self.primary
            h0, gamma = key
            logA = (mus[key] + (Ls[key] @ rng.standard_normal(npts) if "A_field" in sources else 0.0)
                    + ((rng.normal(0, self.disc_split["independent_sd_log"], npts) + rng.normal(0, self.disc_split["common_sd_log"]))
                       if "disc" in sources else 0.0))
            b = bed + (rng.normal(0, BED_SD) if "bed" in sources else 0.0)
            st = stage + (rng.normal(0, STAGE_SD) if "stage" in sources else 0.0)
            z0 = float(np.exp(rng.uniform(np.log(Z0_RANGE[0]), np.log(Z0_RANGE[1])))) if "z0" in sources else self.z0
            h = np.clip(st - b, 0.0, None)
            a = np.exp(logA) * stage_factor(st, h0, gamma) * (np.log(Z_MEAS / self.z0) / np.log(Z_MEAS / z0))
            wet = h > np.e * z0
            out["h"][k] = h
            out["vbar"][k] = np.where(wet, a * (np.log(np.clip(h, np.e * z0, None) / z0) - 1), 0.0)
        return out


def section_mean(samples):
    """Per draw: the plain (unweighted) mean depth-averaged velocity over the points of one section deeper than 2 cm.
    Used only to compare the size of each uncertainty source (step 4 attribution)."""
    vb, h = samples["vbar"], samples["h"]
    wet = h > 0.02
    return np.array([np.mean(vb[k][wet[k]]) if wet[k].any() else np.nan for k in range(len(vb))])


def width_weights(pts):
    """Each point represents the across-channel interval to the midpoints with its neighbours."""
    y = np.asarray(pts.across_m, float)
    order = np.argsort(y)
    w = np.zeros(len(y))
    if len(y) == 1:
        return np.ones(1)
    dy = np.diff(y[order])
    w[order] = np.r_[dy[0] / 2, (dy[:-1] + dy[1:]) / 2, dy[-1] / 2]
    return w


def length_weights(sections):
    """Each section represents the along-reach interval to the midpoints with its neighbours (sums to one)."""
    names = list(sections)
    al = np.array([float(sections[t].along_m.median()) for t in names])
    o = np.argsort(al)
    ala = al[o]
    edges = np.r_[ala[0] - (ala[1] - ala[0]) / 2, (ala[1:] + ala[:-1]) / 2, ala[-1] + (ala[-1] - ala[-2]) / 2]
    lw = np.zeros(len(names))
    lw[o] = np.diff(edges)
    return lw / lw.sum()


def sample_reach(chain, stages, sections, n=200, sources=ALL_SOURCES, seed=0, chunk=400):
    """Joint draws of the whole reach through a sequence of stages: one GP draw over every point of every section,
    one stage error, one z0 and one day-level discrepancy per draw. Returns, per section, (n, T) arrays of the
    width-weighted mean depth-averaged velocity."""
    rng = np.random.default_rng(seed)
    names = list(sections)
    pts = pd.concat([sections[t].assign(_sec=t) for t in names], ignore_index=True)
    npts, T = len(pts), len(stages)
    stages = np.asarray(stages, float)
    disc = chain.disc_split
    keys = [chain.ridge[i] for i in rng.integers(len(chain.ridge), size=n)] if "law" in sources else [chain.primary] * n
    mus, Ls = {}, {}
    for key in set(keys):
        mu, cov = chain.models[key].predict_cov(pts)
        mus[key], Ls[key] = mu, linalg.cholesky(cov + 1e-8 * np.eye(npts), lower=True)
    logA = np.stack([mus[k] + (Ls[k] @ rng.standard_normal(npts) if "A_field" in sources else 0.0) for k in keys])
    if "disc" in sources:
        logA += rng.normal(0, disc["independent_sd_log"], (n, npts)) + rng.normal(0, disc["common_sd_log"], (n, 1))
    bed = pts.bed_m.values[None, :] + (rng.normal(0, BED_SD, (n, npts)) if "bed" in sources else 0.0)
    z0 = (np.exp(rng.uniform(np.log(Z0_RANGE[0]), np.log(Z0_RANGE[1]), n)) if "z0" in sources
          else np.full(n, chain.z0))
    A = np.exp(logA) * (np.log(Z_MEAS / chain.z0) / np.log(Z_MEAS / z0))[:, None]   # a was fitted with the best z0
    h0 = np.array([k[0] for k in keys])
    gam = np.array([k[1] for k in keys])
    sec_idx = {t: np.where(pts._sec.values == t)[0] for t in names}
    ww = {t: width_weights(sections[t]) for t in names}
    out = {t: np.full((n, T), np.nan) for t in names}
    for j0 in range(0, T, chunk):
        js = np.arange(j0, min(j0 + chunk, T))
        st = stages[js][None, :] + (rng.normal(0, STAGE_SD, (n, 1)) if "stage" in sources else 0.0)
        g = stage_factor(st, h0[:, None], gam[:, None])
        for t in names:
            ix = sec_idx[t]
            h = np.clip(st[:, :, None] - bed[:, None, ix], 0, None)
            a = A[:, None, ix] * g[:, :, None]
            wet = h > np.e * z0[:, None, None]
            v = np.where(wet, a * (np.log(np.clip(h, 1e-6, None) / z0[:, None, None]) - 1), 0.0)
            w = ww[t][None, None, :] * wet
            wsum = w.sum(2)
            out[t][:, js] = np.where(wsum > 0, (v * w).sum(2) / np.maximum(wsum, 1e-12), np.nan)
    return out
