"""
roofs.py - roof primitives for one building part, fitted to LiDAR and selected by a penalised robust score.

A primitive is a set of planes combined by `min` (every convex roof: flat, shed, gable, hip, pyramid) or a smooth
surface (dome, cone) tessellated as many planes. Because the roof is the lower envelope of planes, each plane's
face is the part polygon clipped to where that plane is lowest, so ridges and hips are straight lines where the
planes meet, exactly.

Model frame: feet, +x east, +z south, y up (relative to PAD). Planes are (a, b, c) with y = a*x + b*z + c.
All selection is algorithmic: the same candidates, fit and score for every part of every building.
"""
import math
import numpy as np
from shapely.geometry import Point

HUBER = 1.0          # ft: Huber threshold for robust fitting
MIN_PTS = 12


def frame(poly):
    """Long-axis frame of a polygon: origin at its minimum-rotated-rectangle centre, unit vectors along (ux) and
    across (vx) the longer side, half-length L and half-width W."""
    co = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = co[1] - co[0], co[2] - co[1]
    if np.hypot(*e1) < np.hypot(*e2): e1, e2 = e2, e1
    ux = e1 / (np.hypot(*e1) or 1); vx = np.array([-ux[1], ux[0]])
    cen = co.mean(0)
    return cen, ux, vx, np.hypot(*e1) / 2, np.hypot(*e2) / 2


def _plane_from_frame(cen, ux, vx, du, dv, c0):
    """y = c0 + du*u + dv*v with u, v the frame coordinates -> world plane (a, b, c)."""
    a = du * ux[0] + dv * vx[0]; b = du * ux[1] + dv * vx[1]
    c = c0 - a * cen[0] - b * cen[1]
    return (a, b, c)


def huber_lstsq(A, y, iters=8):
    w = np.ones(len(y)); p = None
    for _ in range(iters):
        sw = np.sqrt(w)
        p, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = np.abs(A @ p - y)
        w = np.where(r <= HUBER, 1.0, HUBER / np.maximum(r, 1e-9))
    return p


def eval_planes(planes, x, z):
    P = np.asarray(planes)
    return (P[:, 0][None] * np.asarray(x)[:, None] + P[:, 1][None] * np.asarray(z)[:, None] + P[:, 2][None]).min(1)


class Fit:
    def __init__(self, kind, planes, k, smooth=None):
        self.kind, self.planes, self.k, self.smooth = kind, planes, k, smooth

    def h(self, x, z):
        if self.smooth is not None: return self.smooth(np.asarray(x, float), np.asarray(z, float))
        return eval_planes(self.planes, np.asarray(x, float), np.asarray(z, float))


def _fit_flat(pts):
    c = float(np.median(pts[:, 1]))
    return Fit('flat', [(0.0, 0.0, c)], 1)


def _fit_shed(pts):
    A = np.c_[pts[:, 0], pts[:, 2], np.ones(len(pts))]
    a, b, c = huber_lstsq(A, pts[:, 1])
    return Fit('shed', [(a, b, c)], 3)


def _fit_gable(pts, cen, ux, vx, axis):
    """Two planes sloping away from a ridge parallel to `axis` (0: ridge along the long side, 1: across).
    y = top - s * |w - r| in the frame coordinate w across the ridge; ridge offset r searched, slopes fitted."""
    d = pts[:, [0, 2]] - cen
    w = d @ (vx if axis == 0 else ux)
    best = None
    lo, hi = np.percentile(w, [15, 85])
    for r in np.linspace(lo, hi, 15):
        left = w < r
        if left.sum() < 4 or (~left).sum() < 4: continue
        # y = top + s1*(w - r) on the left (s1 > 0), y = top - s2*(w - r) on the right
        A = np.c_[np.ones(len(w)), np.where(left, w - r, 0), np.where(left, 0, -(w - r))]
        top, s1, s2 = huber_lstsq(A, pts[:, 1])
        if s1 <= 0.05 or s2 <= 0.05: continue
        res = np.abs(A @ np.array([top, s1, s2]) - pts[:, 1])
        sc = np.median(res)
        if best is None or sc < best[0]: best = (sc, r, top, s1, s2)
    if best is None: return None
    _, r, top, s1, s2 = best
    a_ = vx if axis == 0 else ux
    # plane 1: y = top + s1*(w - r); plane 2: y = top - s2*(w - r); roof = min of the two
    p1 = _plane_from_frame(cen, ux, vx, s1 * (a_ @ ux), s1 * (a_ @ vx), top - s1 * r)
    p2 = _plane_from_frame(cen, ux, vx, -s2 * (a_ @ ux), -s2 * (a_ @ vx), top + s2 * r)
    return Fit('gable', [p1, p2], 4)


def _fit_hip(pts, cen, ux, vx, L, W):
    """Four planes rising from the part's rectangle edges: y = eave + min(s*(W - |v|), t*(L - |u|)),
    each side's offset fitted, so ridge length and position follow the data."""
    d = pts[:, [0, 2]] - cen
    u, v = d @ ux, d @ vx
    best = None
    for su in (0.0,):
        # model: y = e + min(s*(W - |v - rv|), t*(L - |u - ru|)) solved by alternating assignment
        rv, ru = float(np.median(v)), float(np.median(u))
        s, t, e = 0.5, 0.5, float(np.percentile(pts[:, 1], 10))
        for _ in range(8):
            fv = W - np.abs(v - rv); fu = L - np.abs(u - ru)
            side = s * fv <= t * fu
            A = np.c_[np.ones(len(u)), np.where(side, fv, 0), np.where(side, 0, fu)]
            e, s, t = huber_lstsq(A, pts[:, 1])
            if s <= 0.05 or t <= 0.05: return None
        res = np.median(np.abs(A @ np.array([e, s, t]) - pts[:, 1]))
        best = (res, e, s, t, rv, ru)
    _, e, s, t, rv, ru = best
    planes = []
    for sg in (1, -1):    # long sides (slope s across v)
        planes.append(_plane_from_frame(cen, ux, vx, 0.0, -sg * s, e + s * W + sg * s * rv))
    for sg in (1, -1):    # hip ends (slope t along u)
        planes.append(_plane_from_frame(cen, ux, vx, -sg * t, 0.0, e + t * L + sg * t * ru))
    return Fit('hip', planes, 5)


def _fit_dome(pts, poly):
    """Round parts (rotundas, towers): y = base + H*sqrt(1 - (r/R)^2) about the part centre (k = 3)."""
    c = poly.centroid; R = math.sqrt(poly.area / math.pi)
    r = np.hypot(pts[:, 0] - c.x, pts[:, 2] - c.y) / R
    q = np.sqrt(np.clip(1 - np.clip(r, 0, 1) ** 2, 0, 1))
    A = np.c_[np.ones(len(r)), q]
    base, H = huber_lstsq(A, pts[:, 1])
    if H <= 0.5: return None
    cx, cz = c.x, c.y
    f = lambda x, z: base + H * np.sqrt(np.clip(1 - np.clip(np.hypot(x - cx, z - cz) / R, 0, 1) ** 2, 0, 1))
    return Fit('dome', [], 3, smooth=f)


def _fit_skeleton(pts, poly, eaves):
    """Hipped roof over the part's own outline: every eave edge (outline edges on the building exterior) carries a
    plane rising inward at one common slope; capped by a flat top. For a convex part this lower envelope is the
    straight-skeleton roof: pyramids, polygonal and conical roofs, irregular hips. k = 3 (eave, slope, top)."""
    if not eaves or poly.area / poly.convex_hull.area < 0.93: return None
    E = []
    for (x0, z0), (x1, z1) in eaves:
        L = math.hypot(x1 - x0, z1 - z0)
        if L < 0.5: continue
        nx, nz = -(z1 - z0) / L, (x1 - x0) / L
        mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
        if not poly.buffer(0.01).contains(Point(mx + nx * 0.3, mz + nz * 0.3)): nx, nz = -nx, -nz
        E.append((nx, nz, -(nx * x0 + nz * z0)))       # inward distance = nx*x + nz*z + d
    if len(E) < 3: return None
    Ea = np.asarray(E)
    dist = (Ea[:, 0][None] * pts[:, [0]] + Ea[:, 1][None] * pts[:, [2]] + Ea[:, 2][None]).min(1)
    top = float(np.percentile(pts[:, 1], 95))
    m = pts[:, 1] < top - 0.5
    if m.sum() < 8: return None
    e, s_ = huber_lstsq(np.c_[np.ones(m.sum()), dist[m]], pts[m, 1])
    if s_ <= 0.08: return None
    planes = [(s_ * a, s_ * b, e + s_ * d) for a, b, d in E] + [(0.0, 0.0, top)]
    return Fit('hipped', planes, 3)


def candidates(pts, poly, eaves=None):
    cen, ux, vx, L, W = frame(poly)
    out = [_fit_flat(pts), _fit_shed(pts)]
    sk = _fit_skeleton(pts, poly, eaves)
    if sk: out.append(sk)
    for ax in (0, 1):
        g = _fit_gable(pts, cen, ux, vx, ax)
        if g: out.append(g)
    hp = _fit_hip(pts, cen, ux, vx, L, W)
    if hp: out.append(hp)
    circ = 4 * math.pi * poly.area / poly.length ** 2
    if circ > 0.8:
        dm = _fit_dome(pts, poly)
        if dm: out.append(dm)
    return out


def coherent_share(pts, poly, cell=2.0):
    """Largest connected patch of the given points (2 ft cells, 8-connected), as a share of the part's area."""
    from scipy import ndimage
    x0, z0, x1, z1 = poly.bounds
    C = int((x1 - x0) / cell) + 2; R = int((z1 - z0) / cell) + 2
    M = np.zeros((R, C), bool)
    M[np.clip(((pts[:, 2] - z0) / cell).astype(int), 0, R - 1), np.clip(((pts[:, 0] - x0) / cell).astype(int), 0, C - 1)] = True
    lab, n = ndimage.label(M, structure=np.ones((3, 3)))
    if n == 0: return 0.0
    return float(np.bincount(lab.ravel())[1:].max() * cell * cell / max(poly.area, 1.0))


def score_of(med, k, n):
    return n * math.log(med ** 2 + 0.01) + k * math.log(max(n, 2)) * 2


def select(pts, poly, gate_median=1.0, gate_p90=3.0, eaves=None):
    """Pick the candidate with the lowest penalised robust score, n*log(MAD^2 + eps) + k*log(n)*2.
    Returns (fit, stats) with stats = {kind, median_abs, p90_abs, n, alternatives}. If even the best candidate
    misses gate_median, returns an 'envelope' (flat at the robust median) and says so: an honest coarse part
    beats a detailed wrong one."""
    n = len(pts)
    if n < MIN_PTS:
        return None, dict(kind='insufficient', n=int(n))
    scored = []
    for f in candidates(pts, poly, eaves):
        d = pts[:, 1] - f.h(pts[:, 0], pts[:, 2])         # + : point above the roof surface
        r = np.abs(d)
        med = float(np.median(r))
        score = score_of(med, f.k, n)
        # Points far ABOVE a roof are tolerated only as scattered clutter (equipment, chimneys, overhanging branches).
        # A coherent raised area is a real higher roof that this candidate misses, so the candidate fails.
        above = d > 3.0
        coh = coherent_share(pts[above], poly) if above.any() else 0.0
        below = np.maximum(0, -d)
        clutter_ok = above.mean() <= 0.3 and coh <= 0.12
        p90 = float(np.percentile(below, 90)) if clutter_ok else float(np.percentile(r, 90))
        ok = med <= gate_median and p90 <= gate_p90
        scored.append(dict(score=score, fit=f, med=med, p90=p90, ok=ok,
                           tails=(float(np.percentile(d, 10)), float(np.percentile(d, 90))), above=float(above.mean()), coherent=coh))
    scored.sort(key=lambda t: t['score'])
    alts = [dict(kind=t['fit'].kind, median_abs=round(t['med'], 2), passes=t['ok']) for t in scored[:5]]
    passing = [t for t in scored if t['ok']]
    best = scored[0]
    if not passing:
        env = _fit_flat(pts); env.kind = 'envelope'
        dd = pts[:, 1] - env.h(pts[:, 0], pts[:, 2]); r = np.abs(dd)
        return env, dict(kind='envelope', reason=f"none of the implemented shapes passed (best {best['fit'].kind}: median {best['med']:.2f} ft, p90 {best['p90']:.2f} ft; gate {gate_median}/{gate_p90})",
                         median_abs=round(float(np.median(r)), 2), p90_abs=round(float(np.percentile(r, 90)), 2),
                         signed_p10=round(float(np.percentile(dd, 10)), 2), signed_p90=round(float(np.percentile(dd, 90)), 2),
                         n=int(n), alternatives=alts, best=dict(kind=best['fit'].kind, score=best['score']))
    t = passing[0]                                          # the best-scoring candidate that PASSES (R2)
    f, med, p90, sc = t['fit'], t['med'], t['p90'], t['score']
    return f, dict(kind=f.kind, median_abs=round(med, 2), p90_abs=round(p90, 2), n=int(n), alternatives=alts, score=sc,
                   signed_p10=round(t['tails'][0], 2), signed_p90=round(t['tails'][1], 2), above_share=round(t['above'], 3), coherent_above=round(t['coherent'], 3))
