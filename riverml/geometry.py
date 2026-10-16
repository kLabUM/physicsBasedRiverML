"""River geometry built from the centerline network and the survey itself.

Everything here works in the survey's horizontal frame, NAD83 StatePlane Michigan South
(ESRI:102690, feet), and converts to metres only where results leave this module.

  1. main_stem()          one continuous centerline through the reach, merged from the branched
                          river network (main channel, tributaries, island loops);
  2. reach_coordinates()  each point's distance along that centerline (m downstream of an origin)
                          and its offset across it (m, positive toward the left bank): the model's frame;
  3. bank_lines()         the two edges of the wetted channel and the channel polygon, placed from the
                          surveyed bank points and, where none were surveyed, from the outermost survey
                          points plus the margin measured where both exist. It works internally in
                          station_offset() / to_map(), a feet-based frame along the line's own direction.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge

from .config import CRS, FT


# ------------------------------------------------------------------ 1. main stem
def main_stem(network_path, easting, northing, buffer_ft=500.0, tol_ft=1.0):
    """Merge the river network around the survey into ONE continuous LineString.

    The network is clipped to the survey's bounding box (plus a buffer). Its parts are
    merged where they meet; starting from the part nearest the most survey points, the
    chain is extended through each junction along the branch that carries the most survey
    points (ties: the longer branch). A single line is needed because projecting onto a
    disconnected MultiLineString gives distances in an arbitrary order."""
    net = gpd.read_file(network_path).to_crs(CRS)
    pts = [Point(e, n) for e, n in zip(easting, northing)]
    minx, miny = np.min(easting), np.min(northing)
    maxx, maxy = np.max(easting), np.max(northing)
    clip = box(minx - buffer_ft, miny - buffer_ft, maxx + buffer_ft, maxy + buffer_ft)
    lines = net[net.geometry.intersects(clip)].geometry.union_all()
    merged = linemerge(lines) if lines.geom_type == "MultiLineString" else lines
    if merged.geom_type == "LineString":
        return merged
    parts = list(merged.geoms)

    counts = np.zeros(len(parts), int)                    # survey points nearest each part
    for p in pts:
        counts[int(np.argmin([p.distance(q) for q in parts]))] += 1

    def touches(a, b):
        return abs(a[0] - b[0]) <= tol_ft and abs(a[1] - b[1]) <= tol_ft

    seed = int(np.argmax(counts))
    used, chain = {seed}, [list(parts[seed].coords)]

    def next_part(node):
        best, best_j = None, None
        for j, p in enumerate(parts):
            if j in used:
                continue
            c = list(p.coords)
            if touches(c[0], node) or touches(c[-1], node):
                key = (counts[j], p.length)
                if best is None or key > best:
                    best, best_j = key, j
        return best_j

    for forward in (True, False):                         # extend from the end, then from the start
        while True:
            node = chain[-1][-1] if forward else chain[0][0]
            j = next_part(node)
            if j is None:
                break
            used.add(j)
            c = list(parts[j].coords)
            if forward:
                chain.append(c[::-1] if touches(c[-1], node) else c)
            else:
                chain.insert(0, c[::-1] if touches(c[0], node) else c)
    coords = []
    for c in chain:
        coords.extend(c[1:] if coords else c)
    return LineString(coords)


# ------------------------------------------------------------------ 2. reach coordinates
def reach_coordinates(line, easting, northing, origin, half_ft=5.0):
    """The model's frame, in metres: distance along the centerline downstream of `origin` (an (E, N) point), and
    offset across it, positive toward the left bank looking downstream. Downstream is taken as the direction in
    which northing decreases (this reach flows south toward Ann Arbor)."""
    s0 = line.project(Point(*origin))
    sign = -1.0 if line.coords[-1][1] > line.coords[0][1] else 1.0   # +1 when the line itself runs downstream
    along, across = [], []
    for e, n in zip(easting, northing):
        s = line.project(Point(e, n))
        q = line.interpolate(s)
        t = (np.array(line.interpolate(min(s + half_ft, line.length)).coords[0])
             - np.array(line.interpolate(max(s - half_ft, 0)).coords[0]))
        t /= np.linalg.norm(t)                                         # local direction of the line
        along.append((s - s0) * FT * sign)
        across.append(float(t[0] * (n - q.y) - t[1] * (e - q.x)) * FT * sign)
    return np.array(along), np.array(across)


# ------------------------------------------------------------------ the frame used to draw the banks
def station_offset(line, easting, northing, eps_ft=1.0):
    """Distance along `line` and signed offset from it, both in feet, for each point.
    Offset sign: negative to the LEFT of the line's direction of travel, positive to the RIGHT."""
    along, offset = [], []
    for e, n in zip(easting, northing):
        p = Point(e, n)
        d = line.project(p)
        p1 = line.interpolate(max(d - eps_ft, 0.0))
        p2 = line.interpolate(min(d + eps_ft, line.length))
        cross = (p2.x - p1.x) * (n - p1.y) - (p2.y - p1.y) * (e - p1.x)
        dist = p.distance(line)
        along.append(d)
        offset.append(-dist if cross > 0 else dist)
    return np.array(along), np.array(offset)


def to_map(line, along, offset, eps_ft=1.0):
    """Inverse of station_offset: (distance along, signed offset) back to (Easting, Northing).
    A positive offset lies to the right of the direction of travel."""
    along = np.clip(np.atleast_1d(np.asarray(along, float)), 0.0, line.length)
    offset = np.atleast_1d(np.asarray(offset, float))
    east, north = np.empty(len(along)), np.empty(len(along))
    for i, (s, o) in enumerate(zip(along, offset)):
        here = line.interpolate(s)
        back, ahead = line.interpolate(max(s - eps_ft, 0.0)), line.interpolate(min(s + eps_ft, line.length))
        dx, dy = ahead.x - back.x, ahead.y - back.y
        d = np.hypot(dx, dy) or 1.0
        nx, ny = dy / d, -dx / d                          # tangent rotated -90 degrees: right of travel
        east[i], north[i] = here.x + nx * o, here.y + ny * o
    return east, north


# a known-answer check on the sign convention: +10 ft from an east-flowing line is south of it
_e, _n = to_map(LineString([(0.0, 0.0), (100.0, 0.0)]), [50.0], [10.0])
assert abs(_e[0] - 50.0) < 1e-6 and _n[0] < -9.99, "reach-frame sign convention is mirrored"


# ------------------------------------------------------------------ 3. bank lines
BANK_NOTES = ["bank height", "edge of bank"]   # field-note wordings that mark a point on the bank
MIN_WET_POINTS = 4        # in-water points a transect needs to place a bank point on it
CALIB_LIMIT_FT = 150.0    # in-water points farther than this from the centerline are not used to place banks
OUTSIDE_TOL_FT = 1.0      # a bank point lying more than this inside the survey on its own side is rejected
GRID_FT = 50.0            # spacing of the stations the edges are defined at
BIN_FT = 100.0            # length of reach summarised by each survey-edge sample
WINDOW_FT = 200.0         # a station counts as surveyed if data lie this close
BANK_REACH_FT = 400.0     # how far along the reach a surveyed bank point is trusted
SMOOTH_FT = 300.0         # moving-average length applied to each edge (91 m)
SAFETY_FT = 2.0           # keep each edge at least this far outside the survey (0.6 m)
MIN_WIDTH_FT = 10.0       # never let the two edges meet
MAX_GAP_FT = 800.0        # break the polygon across a stretch with no data at all
MAX_PASSES = 12           # attempts to pull the edge outside every survey point on the ground


def bank_lines(line, wet, banks, log=print):
    """Edges of the wetted channel and the channel polygon.

    wet:   in-water survey points, columns Easting, Northing, transect, date
    banks: points noted in the field as lying on the bank, columns Easting, Northing, transect, date
    Returns (left LineString, right LineString, Polygon, summary dict)."""
    wet = wet.dropna(subset=["Easting", "Northing"]).copy()
    wet["s"], wet["n"] = station_offset(line, wet.Easting.values, wet.Northing.values)
    s_w, n_w = wet.s.values, wet.n.values

    # (a) place each bank point on its transect: along a transect the offset from the centerline
    #     changes linearly with position, so a fit through the in-water points gives the bank's offset
    placed, rejected = [], []
    pool = wet[wet.n.abs() <= CALIB_LIMIT_FT]
    for (tid, day), pts in banks.groupby(["transect", "date"]):
        t = pool[(pool.transect == tid) & (pool.date == day)].drop_duplicates(["Easting", "Northing"])
        if len(t) < MIN_WET_POINTS:
            continue
        xy = t[["Easting", "Northing"]].values.astype(float)
        mid = xy.mean(0)
        direction = np.linalg.svd(xy - mid)[2][0]          # the line the transect runs along
        u = (xy - mid) @ direction
        if np.ptp(u) < 1.0:
            continue
        slope, icept = np.polyfit(u, t.n.values, 1)
        lo, hi = t.n.min(), t.n.max()                       # outermost in-water point at each end
        for off in slope * ((pts[["Easting", "Northing"]].values.astype(float) - mid) @ direction) + icept:
            past_lo, past_hi = lo - off, off - hi
            if max(past_lo, past_hi) < -OUTSIDE_TOL_FT:     # inside the survey on its own side: a mislabelled note
                rejected.append(tid)
                continue
            placed.append({"side": "left" if past_lo > past_hi else "right",
                           "station": float(t.s.median()), "offset": float(off)})
    P = pd.DataFrame(placed)
    log(f"  bank points: {len(banks)} noted, {len(P)} placed, {len(rejected)} rejected as inside the survey")

    # (b) each edge as a curve of offset against distance along the reach
    grid = np.arange(s_w.min(), s_w.max() + GRID_FT, GRID_FT)
    cuts = np.arange(s_w.min(), s_w.max() + BIN_FT, BIN_FT)
    which = np.clip(np.digitize(s_w, cuts) - 1, 0, len(cuts) - 1)
    b_s = [s_w[which == b].mean() for b in np.unique(which)]
    survey_edge = {"left": np.interp(grid, b_s, [n_w[which == b].min() for b in np.unique(which)]),
                   "right": np.interp(grid, b_s, [n_w[which == b].max() for b in np.unique(which)])}
    has_data = np.min(np.abs(grid[:, None] - s_w[None, :]), axis=1) <= WINDOW_FT
    k = int(round(SMOOTH_FT / GRID_FT))
    curves, margins = {}, {}
    for side, outward in (("left", -1.0), ("right", 1.0)):
        q = P[P.side == side].sort_values("station") if len(P) else P
        edge = survey_edge[side]
        if len(q) >= 2:
            near = np.min(np.abs(grid[:, None] - q.station.values[None, :]), axis=1) <= BANK_REACH_FT
            measured = np.where(near, np.interp(grid, q.station.values, q.offset.values), np.nan)
            ok = np.isfinite(measured)
            margin = max(float(np.median(outward * (measured[ok] - edge[ok]))), 0.0)
        else:
            measured, margin = np.full(grid.shape, np.nan), 10.0
        margins[side] = margin
        curve = np.where(np.isfinite(measured), measured, edge + outward * margin)
        curve = np.convolve(curve, np.ones(k), "same") / np.convolve(np.ones_like(curve), np.ones(k), "same")
        limit = edge + outward * SAFETY_FT
        curves[side] = np.where(outward * (limit - curve) > 0, limit, curve)
    narrow = (curves["right"] - curves["left"]) < MIN_WIDTH_FT
    mid = 0.5 * (curves["left"] + curves["right"])
    curves["left"] = np.where(narrow, mid - MIN_WIDTH_FT / 2, curves["left"])
    curves["right"] = np.where(narrow, mid + MIN_WIDTH_FT / 2, curves["right"])

    # (c) close the edges into a polygon and push them out until every survey point is inside
    gaps = np.flatnonzero(np.diff(grid[has_data]) > MAX_GAP_FT) + 1
    blocks = [b for b in np.split(np.flatnonzero(has_data), gaps) if len(b) >= 2]
    idx = max(blocks, key=len)                              # the reach is one continuous stretch
    pts = gpd.GeoSeries(gpd.points_from_xy(wet.Easting, wet.Northing), crs=CRS)
    for _ in range(MAX_PASSES):
        left = LineString(np.column_stack(to_map(line, grid[idx], curves["left"][idx])))
        right = LineString(np.column_stack(to_map(line, grid[idx], curves["right"][idx])))
        poly = Polygon(np.vstack([np.array(left.coords), np.array(right.coords)[::-1]]))
        poly = poly if poly.is_valid else poly.buffer(0)
        outside = ~pts.within(poly.buffer(0.01))
        if not outside.any():
            break
        push = pts[outside].distance(poly) + 0.5
        need = {"left": np.zeros(grid.shape), "right": np.zeros(grid.shape)}
        for ps, pn, dist in zip(s_w[outside.values], n_w[outside.values], push.values):
            i = int(np.argmin(np.abs(grid - ps)))
            side = "left" if abs(pn - curves["left"][i]) <= abs(pn - curves["right"][i]) else "right"
            w = np.abs(grid - ps) <= WINDOW_FT
            need[side][w] = np.maximum(need[side][w], dist)
        curves["left"] -= need["left"]
        curves["right"] += need["right"]
    width_ft = (curves["right"] - curves["left"])[idx]
    summary = {"bank_points_placed": int(len(P)), "bank_points_rejected": len(rejected),
               "margin_m": {s: round(m * FT, 2) for s, m in margins.items()},
               "median_width_m": round(float(np.median(width_ft)) * FT, 1),
               "area_ha": round(poly.area * FT ** 2 / 1e4, 2),
               "survey_points_outside": int(outside.sum())}
    return left, right, poly, summary
