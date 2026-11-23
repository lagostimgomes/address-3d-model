"""
solids.py - a building as explicit parts (solids) with one consistent exterior.

1. Withhold spatial blocks of roof points (validation, C4) BEFORE anything is segmented or fitted.
2. Partition the footprint into parts where the measured roof height steps (a taller wing, a drum, a tower
   setback). Part outlines come from one shared raster partition, simplified as a coverage, so neighbouring
   parts share exactly the same boundary; then clipped to the footprint (arbitrary polygon, holes and curves kept).
3. Fit and select a roof primitive per part (roofs.py); envelope when no primitive passes.
4. Mesh: roof faces = part polygon clipped to where each plane is lowest (straight ridges); walls on every part
   edge, from the lower neighbour's roof (or the ground) up to this part's roof, so upper walls always exist and
   internal faces never do. All rings share one vertex set, so there are no T-junctions.
5. Openings are laid out on the final exposed wall bands only.
"""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box as sbox
from shapely.ops import unary_union
import roofs as RF

GRID = 0.05          # ft: part and face geometry snap to this grid (merges near-duplicate vertices from clipping)
STEP_FT = 3.5        # a height jump larger than this between neighbouring cells is a part boundary
MIN_PART_FT2 = 120   # smaller regions join a neighbour
HOLDOUT_BLOCK = 10.0  # ft: withheld blocks for held-out validation
HOLDOUT_SHARE = 0.10


def polys(g):
    if g is None or g.is_empty: return []
    if isinstance(g, Polygon): return [g]
    return [p for p in getattr(g, 'geoms', []) if isinstance(p, Polygon) and p.area > 1e-6]


def holdout_mask(x, z, seed):
    """Deterministic spatial blocks: about 10% of 10 ft blocks are withheld (C4: before segmentation/fitting)."""
    bx = np.floor(x / HOLDOUT_BLOCK).astype(np.int64); bz = np.floor(z / HOLDOUT_BLOCK).astype(np.int64)
    h = (bx * 73856093) ^ (bz * 19349663) ^ (seed * 83492791)
    return (h % 1000) < int(HOLDOUT_SHARE * 1000)


def height_raster(pg, P, cell):
    """Robust roof height per cell (median of single-echo returns; lower quartile of multi-echo where no single
    echo) on a grid aligned with the footprint's long axis. Returns grid geometry and the raster."""
    cen, ux, vx, L, W = RF.frame(pg)
    d = P[:, [0, 2]] - cen; u = d @ ux; v = d @ vx
    B = np.asarray(pg.exterior.coords) - cen
    u0, u1 = (B @ ux).min() - cell, (B @ ux).max() + cell; v0, v1 = (B @ vx).min() - cell, (B @ vx).max() + cell
    C = int(math.ceil((u1 - u0) / cell)); R = int(math.ceil((v1 - v0) / cell))
    j = np.clip(((u - u0) / cell).astype(int), 0, C - 1); i = np.clip(((v - v0) / cell).astype(int), 0, R - 1)
    H = np.full((R, C), np.nan)
    for sel, q in ((P[:, 3] > 1, 0.25), (P[:, 3] <= 1, 0.5)):      # single echoes overwrite multi
        if not sel.any(): continue
        key = i[sel] * C + j[sel]; ys = P[sel, 1]; o = np.lexsort((ys, key)); key, ys = key[o], ys[o]
        st = np.r_[0, np.nonzero(np.diff(key))[0] + 1]; en = np.r_[st[1:], len(key)]
        H.flat[key[st]] = ys[(st + ((en - st - 1) * q).astype(int))]
    gu = u0 + (np.arange(C) + .5) * cell; gv = v0 + (np.arange(R) + .5) * cell
    GU, GV = np.meshgrid(gu, gv)
    GX = cen[0] + GU * ux[0] + GV * vx[0]; GZ = cen[1] + GU * ux[1] + GV * vx[1]
    inside = shapely.contains_xy(pg.buffer(cell * 0.75), GX, GZ)
    return dict(cen=cen, ux=ux, vx=vx, u0=u0, v0=v0, cell=cell, R=R, C=C, GX=GX, GZ=GZ, inside=inside), H


def partition(pg, G, H):
    """Parts = regions of the height raster with no step > STEP_FT inside them. Returns polygons (a coverage of
    the footprint) and each part's median height."""
    from scipy import ndimage
    ins = G['inside']; have = ins & ~np.isnan(H)
    if have.sum() < 6:
        return [pg], [np.nan]
    # smooth only among measured cells (a nan-aware 3x3 median), never across unmeasured gaps
    Hm = np.where(have, H, np.nan)
    pad = np.pad(Hm, 1, constant_values=np.nan)
    stack = np.stack([pad[1 + a:1 + a + H.shape[0], 1 + b:1 + b + H.shape[1]] for a in (-1, 0, 1) for b in (-1, 0, 1)])
    with np.errstate(all='ignore'):
        Hs = np.where(have, np.nanmedian(stack, axis=0), np.nan)
    R, C = Hs.shape
    # connect 4-neighbours that are BOTH measured and differ by less than STEP_FT
    lab = -np.ones((R, C), int); n = 0
    from collections import deque
    Hf = Hs
    for i0 in range(R):
        for j0 in range(C):
            if not have[i0, j0] or lab[i0, j0] >= 0: continue
            q = deque([(i0, j0)]); lab[i0, j0] = n
            while q:
                i, j = q.popleft()
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < R and 0 <= b < C and have[a, b] and lab[a, b] < 0 and abs(Hf[a, b] - Hf[i, j]) < STEP_FT:
                        lab[a, b] = n; q.append((a, b))
            n += 1
    # unmeasured footprint cells join the nearest labelled cell (labels only; heights are never filled)
    if (ins & (lab < 0)).any() and (lab >= 0).any():
        idx = ndimage.distance_transform_edt(lab < 0, return_distances=False, return_indices=True)
        fill = lab[tuple(idx)]
        lab = np.where(ins & (lab < 0), fill, lab)
    Hf = np.where(have, Hf, np.nan)
    # small regions join the neighbour with the most similar height
    cell2 = G['cell'] ** 2
    for _ in range(4):
        sizes = np.bincount(lab[lab >= 0], minlength=n)
        meds = {k: float(np.nanmedian(Hf[lab == k])) if np.isfinite(Hf[lab == k]).any() else np.nan for k in range(n) if sizes[k]}
        changed = False
        for k in sorted(range(n), key=lambda k: sizes[k]):
            if not sizes[k] or sizes[k] * cell2 >= MIN_PART_FT2: continue
            m = lab == k; ring = ndimage.binary_dilation(m) & ~m & (lab >= 0)
            nb = np.unique(lab[ring]); nb = nb[nb != k]
            if not len(nb): continue
            tgt = min(nb, key=lambda t: abs(meds.get(t, 1e9) - meds[k]) if np.isfinite(meds.get(t, np.nan)) and np.isfinite(meds[k]) else 1e9)
            lab[m] = tgt; changed = True
            sizes = np.bincount(lab[lab >= 0], minlength=n)
        if not changed: break
    ks = [k for k in np.unique(lab) if k >= 0]
    if len(ks) == 1:
        return [pg], [float(np.nanmedian(Hf[ins]))]
    # one polygon per region from its cells, then a consistent simplification of the whole coverage
    c = G['cell']; cen, ux, vx = G['cen'], G['ux'], G['vx']
    def cellpoly(i, j):
        u0 = G['u0'] + j * c; v0 = G['v0'] + i * c
        pts = [(u0, v0), (u0 + c, v0), (u0 + c, v0 + c), (u0, v0 + c)]
        return Polygon([(cen[0] + a * ux[0] + b * vx[0], cen[1] + a * ux[1] + b * vx[1]) for a, b in pts])
    regs = []
    for k in ks:
        cells = [cellpoly(i, j) for i, j in zip(*np.nonzero(lab == k))]
        regs.append(unary_union(cells).buffer(0))
    simp = shapely.coverage_simplify(np.array(regs, dtype=object), tolerance=max(1.0, c * 0.8), simplify_boundary=False)
    parts, meds = [], []
    for k, g in zip(ks, simp):
        for p in polys(g.intersection(pg)):
            if p.area >= 20:
                parts.append(p); meds.append(float(np.nanmedian(Hf[lab == k])) if np.isfinite(Hf[lab == k]).any() else np.nan)
    # slivers lost to clipping: give any uncovered footprint area to the touching part with most shared edge
    rest = pg.difference(unary_union(parts))
    for r in polys(rest):
        if r.area < 1: continue
        k = max(range(len(parts)), key=lambda t: parts[t].intersection(r.buffer(0.05)).area)
        parts[k] = unary_union([parts[k], r]).buffer(0)
    return [max(polys(p), key=lambda q: q.area) for p in parts], meds


def faces_of(part, fit):
    """Roof faces: for min-of-planes, the part clipped to where each plane is lowest; for smooth roofs, a
    constrained triangulation with interior points. Returns [(polygon, plane_index or None)]."""
    if fit.smooth is not None:
        return [(part, None)]
    P = fit.planes
    if len(P) == 1: return [(part, 0)]
    x0, z0, x1, z1 = part.bounds; big = sbox(x0 - 10, z0 - 10, x1 + 10, z1 + 10)
    out = []
    for i, (a, b, c) in enumerate(P):
        reg = part
        for j, (a2, b2, c2) in enumerate(P):
            if i == j: continue
            # half-plane where plane i <= plane j: (a-a2) x + (b-b2) z + (c-c2) <= 0
            da, db, dc = a - a2, b - b2, c - c2
            reg = reg.intersection(halfplane(big, da, db, dc))
            if reg.is_empty: break
        for p in polys(reg):
            if p.area > 0.05: out.append((p, i))
    return out


def halfplane(box_, a, b, c):
    """Polygon {(x, z) in box_: a x + b z + c <= 0}."""
    x0, z0, x1, z1 = box_.bounds
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return box_ if c <= 0 else Polygon()
    corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    vals = [a * x + b * z + c for x, z in corners]
    pts = []
    for k in range(4):
        p, q = corners[k], corners[(k + 1) % 4]; vp, vq = vals[k], vals[(k + 1) % 4]
        if vp <= 0: pts.append(p)
        if (vp < 0 < vq) or (vq < 0 < vp):
            t = vp / (vp - vq); pts.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return Polygon(pts) if len(pts) >= 3 else Polygon()


def clean_partition(parts, pg, spike=0.5, rounds=3):
    """Make the parts a clean coverage of the footprint: thin spikes and stray fragments (from simplification and
    clipping) go to the neighbour sharing the longest boundary with them; every part keeps one piece; all parts
    are snapped to the 0.01 ft grid together so shared boundaries stay identical."""
    from shapely.geometry import JOIN_STYLE
    P = [shapely.set_precision(p, GRID) for p in parts if not p.is_empty]
    for _ in range(rounds):
        moved = False
        for i in range(len(P)):
            if P[i] is None or P[i].is_empty: continue
            core = P[i].buffer(-spike, join_style=JOIN_STYLE.mitre).buffer(spike, join_style=JOIN_STYLE.mitre).intersection(P[i])
            pieces = polys(core)
            keep = max(pieces, key=lambda q: q.area) if pieces else None
            if keep is None or keep.area < 1: continue
            loose = polys(P[i].difference(keep))
            for fr in loose:
                if fr.area < 1e-4: continue
                best, bl = None, 0.0
                for j in range(len(P)):
                    if j == i or P[j] is None or P[j].is_empty: continue
                    l = fr.boundary.intersection(P[j].boundary)
                    l = l.length if l is not None else 0.0
                    if l > bl: best, bl = j, l
                if best is None or bl < 0.05: continue          # on the exterior only: stays with this part
                P[best] = shapely.set_precision(unary_union([P[best], fr]), GRID)
                P[i] = shapely.set_precision(P[i].difference(fr), GRID); moved = True
        if not moved: break
    out = []
    for p in P:
        if p is None: continue
        ps = polys(p)
        if ps: out.append(max(ps, key=lambda q: q.area))
    # remove sub-0.15 ft edges (cut points off the grid) from ALL parts together, so shared boundaries stay identical
    try:
        simp = shapely.coverage_simplify(np.array(out, dtype=object), tolerance=0.15, simplify_boundary=False)
        if all(not g.is_empty for g in simp): out = [max(polys(g), key=lambda q: q.area) if polys(g) else o for g, o in zip(simp, out)]
    except Exception:
        pass
    return [shapely.set_precision(p, GRID) for p in out]


def _pts_in(pp, T):
    inn = shapely.contains_xy(pp, T[:, 0], T[:, 2]) if len(T) else np.zeros(0, bool)     # exact: no neighbour heights
    s = inn & (T[:, 3] <= 1)
    return (T[s] if s.sum() >= RF.MIN_PTS else T[inn])[:, :3]


def eaves_of(pp, pg):
    """Edges of a part that lie on the building's exterior (real eaves), from its simplified outline."""
    ext = pg.boundary.buffer(0.3)
    co = list(pp.simplify(1.0, preserve_topology=True).exterior.coords)
    return [(a, b) for a, b in zip(co[:-1], co[1:]) if ext.contains(LineString([a, b]))]


def _best(pp, T, pg=None):
    """(fit, stats, score) of the best primitive on a part; failing parts score as their best primitive."""
    pts = _pts_in(pp, T)
    fit, st = RF.select(pts, pp, eaves=eaves_of(pp, pg) if pg is not None else None)
    if fit is None: return None, st, None
    sc = st.get('score') if st['kind'] != 'envelope' else st['best']['score']
    return fit, st, sc


def cut_lines_for(pp):
    """Candidate straight cuts: from each reflex corner of the simplified outline, along both of its edges' directions;
    plus cuts across the long axis at 1/3, 1/2, 2/3 and along it at 1/2. Arbitrary outlines keep their own corners."""
    simp = pp.simplify(2.0, preserve_topology=True)
    co = np.asarray(simp.exterior.coords)[:-1]; n = len(co)
    x0, z0, x1, z1 = pp.bounds; big = math.hypot(x1 - x0, z1 - z0) * 2 + 10
    lines = []
    sign = 1 if simp.exterior.is_ccw else -1
    for i in range(n):
        a, b, c = co[i - 1], co[i], co[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        turn = math.degrees(abs(math.atan2(cross, (b - a) @ (c - b))))
        if cross * sign < 0 and turn > 20:            # reflex corner
            for d in (b - a, c - b):
                L = math.hypot(*d)
                if L < 1e-6: continue
                d = d / L
                lines.append(LineString([b - d * big, b + d * big]))
    cen, ux, vx, Lh, Wh = RF.frame(pp)
    for f in (-1 / 3, 0, 1 / 3):
        o = cen + ux * (2 * Lh * f / 1)
        lines.append(LineString([o - vx * big, o + vx * big]))
    lines.append(LineString([cen - ux * big, cen + ux * big]))
    return lines


def refine_part(pp, T, depth=0, max_depth=4, pg=None):
    from shapely.ops import split
    fit, st, sc = _best(pp, T, pg)
    if fit is None or st['kind'] != 'envelope' or depth >= max_depth or pp.area < 2 * MIN_PART_FT2:
        return [pp]
    best = None
    for ln in cut_lines_for(pp):
        try: pieces = [g for g in split(pp, ln).geoms if isinstance(g, Polygon)]
        except Exception: continue
        if len(pieces) < 2 or min(g.area for g in pieces) < MIN_PART_FT2: continue
        tot = 0.0; ok = True
        for g in pieces:
            f2, s2, c2 = _best(g, T, pg)
            if f2 is None: ok = False; break
            tot += c2
        if not ok: continue
        tot += math.log(max(len(_pts_in(pp, T)), 2)) * 6          # a cut must pay for itself
        if best is None or tot < best[0]: best = (tot, pieces)
    if best is None or best[0] >= sc:
        return [pp]
    out = []
    for g in best[1]: out += refine_part(g, T, depth + 1, max_depth, pg)
    return out


# ------------------------------------------------------------------ meshing
Q = 100.0            # encoding: int32 at 0.01 ft (C4)


class Buf:
    def __init__(self):
        self.p, self.uv = [], []

    def tri(self, a, b, c, ta=None, tb=None, tc=None):
        qa, qb, qc = (tuple(np.round(np.asarray(v, float) * Q).astype(np.int64)) for v in (a, b, c))
        if qa == qb or qb == qc or qa == qc: return             # collapsed after quantization (a band tapering to zero)
        cr = np.cross(np.subtract(qb, qa), np.subtract(qc, qa))
        if not cr.any(): return                                  # collinear after quantization
        self.p += [*a, *b, *c]
        if ta is not None: self.uv += [*ta, *tb, *tc]

    def pack(self):
        import base64
        enc = lambda v: base64.b64encode(np.round(np.asarray(v, float) * Q).astype('<i4').tobytes()).decode()
        return dict(p=enc(self.p), uv=enc(self.uv) if self.uv and len(self.uv) * 3 == len(self.p) * 2 else None,
                    n=len(self.p), q=Q)


class VertexSet:
    """All roof-face and part-ring vertices of one building, with points closer than MERGE (left by clipping) merged
    into one representative (an original vertex, so it stays on its lines). Every ring is rewritten through the
    representatives and every edge is split at any representative lying on it: faces and walls meet vertex to
    vertex, with no T-junctions or micro-segments."""
    MERGE = 0.06          # just over the 0.05 ft part grid: distinct boundary vertices are never this close

    def __init__(self, rings):
        pts = [(round(x, 2), round(z, 2)) for r in rings for x, z in r]
        self.P = np.zeros((0, 2)); self.add(pts)

    def add(self, pts):
        from scipy.spatial import cKDTree
        allp = np.unique(np.round(np.vstack([self.P, np.asarray(pts, float).reshape(-1, 2)]), 2), axis=0)
        tree = cKDTree(allp); rep = -np.ones(len(allp), int)
        for i in range(len(allp)):
            if rep[i] >= 0: continue
            for j in tree.query_ball_point(allp[i], self.MERGE): 
                if rep[j] < 0: rep[j] = i
        keep = np.unique(rep)
        self.P = allp[keep]
        self.tree = cKDTree(self.P)

    def snap(self, p):
        d, i = self.tree.query(p)
        return tuple(self.P[i]) if d <= self.MERGE * 1.5 else (round(p[0], 2), round(p[1], 2))

    def split(self, a, b, tol=0.07):
        a = np.asarray(self.snap(a), float); b = np.asarray(self.snap(b), float); d = b - a; L2 = d @ d
        if L2 < 1e-12: return [tuple(a), tuple(b)]
        t = ((self.P - a) @ d) / L2
        proj = a + np.outer(t, d)
        on = (t > 1e-6) & (t < 1 - 1e-6) & (np.hypot(*(self.P - proj).T) < tol)
        idx = np.nonzero(on)[0]; idx = idx[np.argsort(t[idx])]
        out = [tuple(a)] + [tuple(self.P[i]) for i in idx] + [tuple(b)]
        return [p for k, p in enumerate(out) if k == 0 or p != out[k - 1]]


def ring_coords(p):
    return [list(p.exterior.coords)[:-1]] + [list(h.coords)[:-1] for h in p.interiors]


def densify_ring(poly, step):
    return shapely.segmentize(poly, step)


def build(pg, P, terr, seed=0, cut_lines=(), ways=(), nodes=(), roads_xy=None, use='yes', GEO=None):
    """Reconstruct one building. P: Nx4 (x, y, z, n_returns) roof-candidate points inside the footprint.
    Returns dict(mesh=..., parts=[...], windows, doors, validation) or None if there is no usable evidence."""
    test = holdout_mask(P[:, 0], P[:, 2], seed) if len(P) else np.zeros(0, bool)
    T, V = P[~test], P[test]
    area = pg.area
    cell = 1.5 if area < 6000 else (2.0 if area < 30000 else 3.0)
    # adapt the raster to the data: coarsen until ~85% of footprint cells hold a return, so connectivity uses measured
    # cells only without fragmenting sparse surveys (never coarser than 4.5 ft)
    while True:
        G, H = height_raster(pg, T, cell)
        share = float(np.mean(~np.isnan(H[G['inside']]))) if G['inside'].any() else 1.0
        if share >= 0.85 or cell >= 4.5: break
        cell = min(4.5, cell * 1.25)
    G['measured_share'] = share
    # simplest model first: if one primitive over the whole footprint passes the residual gate, the building is one
    # part (a plain gable or hip house, a flat block). Otherwise partition by height steps, then split recursively.
    whole = None
    if len(T):
        sW = T[T[:, 3] <= 1][:, :3]
        uW = sW if len(sW) >= RF.MIN_PTS else T[:, :3]
        fW, stW = RF.select(uW, pg, eaves=eaves_of(pg, pg))
        if fW is not None and stW['kind'] not in ('envelope', 'insufficient'): whole = True
    if whole:
        part_polys, meds = [pg], [float(np.nanmedian(H))] if np.isfinite(H).any() else [np.nan]
    else:
        part_polys, meds = partition(pg, G, H)
        # parts that no single primitive explains are split further by the best straight cut (recursive)
        refined = []
        for pp in part_polys: refined += refine_part(pp, T, pg=pg)
        part_polys = clean_partition(refined, pg)
    def make_part(pp):
        inn = shapely.contains_xy(pp, T[:, 0], T[:, 2]) if len(T) else np.zeros(0, bool)
        pts = T[inn][:, :3]
        sP = T[inn & (T[:, 3] <= 1)][:, :3] if len(T) else pts
        use_pts = sP if len(sP) >= RF.MIN_PTS else pts
        fit, st = RF.select(use_pts, pp, eaves=eaves_of(pp, pg))
        if fit is None:      # no returns at all on this part: an envelope at the neighbouring parts' height
            hh = np.nanmedian([m for m in meds if not np.isnan(m)]) if any(not np.isnan(m) for m in meds) else float(terr(pp.centroid.x, pp.centroid.y)) + 12
            fit = RF.Fit('envelope', [(0.0, 0.0, float(hh))], 1); st = dict(kind='envelope', reason='no returns on part', n=0)
        # held-out residual on this part (points withheld before partition and fitting)
        vin = shapely.contains_xy(pp, V[:, 0], V[:, 2]) if len(V) else np.zeros(0, bool)
        vr = np.abs(fit.h(V[vin, 0], V[vin, 2]) - V[vin, 1]) if vin.any() else np.array([])
        msk = shapely.contains_xy(pp, G['GX'], G['GZ'])
        cov = float(np.mean(~np.isnan(H[msk]))) if msk.any() else None
        st.update(heldout=dict(n=int(len(vr)), median_abs=round(float(np.median(vr)), 2) if len(vr) else None,
                               p90_abs=round(float(np.percentile(vr, 90)), 2) if len(vr) else None, units='ft',
                               underpowered=bool(len(vr) < 50)),
                  coverage=None if cov is None else round(cov, 3), density=round(len(pts) / max(pp.area, 1), 3), area_ft2=round(pp.area))
        med = float(np.median(use_pts[:, 1])) if len(use_pts) else np.nan
        return dict(poly=pp, fit=fit, stats=st, med=med)
    parts = [make_part(pp) for pp in part_polys]
    # a small part no shape explains joins the adjacent part of the most similar height when the joined part then
    # passes the gate (typically a corner under tree canopy); otherwise it stays an envelope
    changed = True
    while changed:
        changed = False
        for i, prt in enumerate(parts):
            if prt['stats']['kind'] != 'envelope' or prt['poly'].area >= 2 * MIN_PART_FT2: continue
            nb = [(j, q) for j, q in enumerate(parts) if j != i and prt['poly'].intersection(q['poly'].buffer(0.05)).length > 0
                  and prt['poly'].buffer(0.05).intersection(q['poly']).area > 0.5]
            nb = [(j, q) for j, q in nb if not (np.isnan(prt['med']) or np.isnan(q['med'])) and abs(prt['med'] - q['med']) <= STEP_FT]
            for j, q in sorted(nb, key=lambda t: abs(prt['med'] - t[1]['med'])):
                u = unary_union([prt['poly'], q['poly']]).buffer(0)
                if u.geom_type != 'Polygon': continue
                m = make_part(u)
                if m['stats']['kind'] in ('envelope', 'insufficient'): continue
                m['stats']['merged_from'] = 'a small unexplained part joined to its neighbour of similar height'
                parts = [x for t, x in enumerate(parts) if t not in (i, j)] + [m]
                changed = True; break
            if changed: break
    # S2.3: an envelope part gets one more chance as a support-constrained planar B-rep; accepted only if every plane
    # region passes the residual gate and the penalised score beats the envelope's
    expanded = []
    for prt in parts:
        st = prt['stats']
        if st['kind'] != 'envelope' or prt['poly'].area < 2 * MIN_PART_FT2:
            expanded.append(prt); continue
        res = brep_split(prt['poly'], T, pg)
        if not res:
            expanded.append(prt); continue
        pts_all = _pts_in(prt['poly'], T)
        n_all = max(len(pts_all), 2)
        env_med = float(np.median(np.abs(pts_all[:, 1] - np.median(pts_all[:, 1])))) if len(pts_all) else 99.0
        env_score = RF.score_of(env_med, 1, n_all)
        subs, total, ok = [], 0.0, True
        for r, pl, n in res:
            rp = _pts_in(r, T)
            if len(rp) < RF.MIN_PTS: ok = False; break
            d = rp[:, 1] - (pl[0] * rp[:, 0] + pl[1] * rp[:, 2] + pl[2])
            med = float(np.median(np.abs(d))); below = np.maximum(0, -d)
            above = d > 3.0
            coh = RF.coherent_share(rp[above], r) if above.any() else 0.0
            p90 = float(np.percentile(below, 90)) if (above.mean() <= 0.3 and coh <= 0.12) else float(np.percentile(np.abs(d), 90))
            if med > 1.0 or p90 > 3.0 or math.hypot(pl[0], pl[1]) > RF.MAX_SLOPE: ok = False; break
            total += RF.score_of(med, 3, len(rp))
            subs.append((r, pl, len(rp), med, p90, float(np.percentile(d, 10)), float(np.percentile(d, 90))))
        if not ok or total + math.log(n_all) * 6 >= env_score:
            prt['stats']['brep'] = 'tried: rejected (a region failed the gate or the score did not beat the envelope)'
            expanded.append(prt); continue
        for r, pl, n, med, p90, t10, t90 in subs:
            f = RF.Fit('planar-brep', [tuple(pl)], 3)
            vin = shapely.contains_xy(r, V[:, 0], V[:, 2]) if len(V) else np.zeros(0, bool)
            expanded.append(dict(poly=r, fit=f, stats=dict(kind='planar-brep', median_abs=round(med, 2), p90_abs=round(p90, 2),
                signed_p10=round(t10, 2), signed_p90=round(t90, 2), n=int(n), area_ft2=round(r.area),
                heldout=dict(n=int(vin.sum()), median_abs=None, p90_abs=None, units='ft', underpowered=bool(vin.sum() < 50)),
                coverage=None, density=round(n / max(r.area, 1), 3),
                reason='one plane of a piecewise-planar roof (support-constrained, joints on plane intersections)')))
    parts = expanded
    mesh, raw, hfun, part_index, other = mesh_parts(parts, terr, cut_lines)
    owner = mesh_parts.owner
    # withheld points for the exported-surface check (gates.py), by FINAL part and spatial block
    heldout = {}
    for x, y, z in V[:, :3] if len(V) else []:
        k = owner(x, z)
        if k is None: continue
        blk = int(math.floor(x / HOLDOUT_BLOCK)) * 100003 + int(math.floor(z / HOLDOUT_BLOCK))
        heldout.setdefault(f'p{k}', []).append([round(float(x), 2), round(float(y), 2), round(float(z), 2), blk])
    return dict(parts=parts, mesh=mesh, raw=raw, hfun=hfun, part_index=part_index, other=other, owner=owner, holdout_n=int(test.sum()), heldout=heldout,
                snap=dict(n=len(mesh_parts.snap_moves), max_ft=max(mesh_parts.snap_moves, default=0.0), units='ft',
                          note='height reconciliation where two parts meet on a crossing or within 0.1 ft of one'))


def mesh_parts(parts, terr, cut_lines=()):
    """Exact meshing through one planar arrangement. All part outlines and roof-face lines (ridges, hips, valleys)
    are noded together and polygonized; each cell belongs to one part and one roof plane. Walls come from the
    arrangement's edges, with the part (or the exterior) on each side. Because every face and wall uses the same
    noded vertices, shared edges are identical by construction: no T-junctions, gaps or slivers."""
    from shapely.ops import polygonize
    face_lines = []
    for prt in parts:
        prt['faces'] = []
        if prt['fit'].smooth is not None:
            prt['poly'] = shapely.segmentize(prt['poly'], 2.0)
            prt['faces'].append((prt['poly'], None))
        else:
            for f, pi in faces_of(prt['poly'], prt['fit']):
                prt['faces'].append((f, pi)); face_lines.append(f.boundary)
    # snap-rounding the whole noded linework as ONE geometry merges near-coincident nodes consistently
    from shapely.ops import snap
    lines = unary_union([p_['poly'].boundary for p_ in parts] + face_lines)
    for _ in range(2):                    # merge nodes closer than 0.12 ft (clipping/cleanup jogs), then re-node
        lines = unary_union(shapely.set_precision(snap(lines, lines, 0.12), 0.01))
    cells = [c for c in polygonize(lines) if c.area > 1e-9]          # tiny cells are merged below, never dropped (a dropped cell is a hole)
    part_index = shapely.STRtree([p_['poly'] for p_ in parts])
    def owner(x, z):
        pt = Point(x, z)
        for j in part_index.query(pt):
            if parts[j]['poly'].contains(pt): return int(j)
        return None
    hcache = {}
    def hfun(k, x, z):
        # evaluate at the exported (0.01 ft) plan position: two computed positions that export to the same vertex must
        # get the same height, or a steep plane turns a few thousandths of a foot into an open seam (review s2-native)
        x, z = round(x, 2), round(z, 2)
        key = (x, z)
        y = round(float(parts[k]['fit'].h(np.array([x]), np.array([z]))[0]), 2)
        for (k2, y2) in hcache.get(key, ()):
            if k2 == k: return y2
            if abs(y2 - y) <= 0.03: y = y2; break        # the same surface seen from two parts: one value
        hcache.setdefault(key, []).append((k, y))
        return y
    other = lambda x, z, k: (lambda j: None if j == k else j)(owner(x, z))
    # cells -> (part, plane)
    cell_info = []
    for c in cells:
        rp = c.representative_point(); k = owner(rp.x, rp.y)
        if k is None:                                     # snapped sliver outside every original part: nearest part
            d = [parts[j]['poly'].distance(rp) for j in range(len(parts))]
            if min(d) > 0.2: continue
            k = int(np.argmin(d))
        pi = None
        if parts[k]['fit'].smooth is None:
            P_ = parts[k]['fit'].planes
            vals = [a_ * rp.x + b_ * rp.y + c_ for a_, b_, c_ in P_]
            pi = int(np.argmin(vals))                     # min-of-planes: the lowest plane owns the cell
        cell_info.append((c, k, pi))
    # sliver cells (near-coincident lines) join the neighbouring cell they share the longest boundary with, so their
    # edges become internal (no wall) instead of producing near-duplicate walls
    if cell_info:
        ctree = shapely.STRtree([c for c, _, _ in cell_info])
        for idx, (c, k, pi) in enumerate(cell_info):
            if c.area >= 0.05: continue
            best, bl = None, 0.0
            for j in ctree.query(c.buffer(0.01)):
                if j == idx or cell_info[j][0].area < 0.05: continue
                l = c.boundary.intersection(cell_info[j][0].boundary)
                l = l.length if l is not None else 0.0
                if l > bl: best, bl = j, l
            if best is not None:
                cell_info[idx] = (c, cell_info[best][1], cell_info[best][2])
    # from here on the arrangement is the only geometry: each part is the union of its cells
    for k in range(len(parts)):
        cs = [c for c, kk, _ in cell_info if kk == k]
        if cs: parts[k]['poly'] = unary_union(cs)
    part_index = shapely.STRtree([p_['poly'] for p_ in parts])
    # arrangement edges with the part on each side, from CELL ADJACENCY (exact; no probes that could jump a sliver).
    # With every cell ring oriented counter-clockwise (holes clockwise), a cell's interior lies left of each directed
    # edge, so its outward normal is the right-hand normal.
    from shapely.geometry.polygon import orient
    edge_cells = {}
    for ci, (c, k, pi) in enumerate(cell_info):
        oc = orient(c, 1.0)
        for ring in [oc.exterior] + list(oc.interiors):
            co = list(ring.coords)
            for p0, p1 in zip(co[:-1], co[1:]):
                if p0 == p1: continue
                L_ = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                out_n = ((p1[1] - p0[1]) / L_, -(p1[0] - p0[0]) / L_)
                key = (p0, p1) if p0 < p1 else (p1, p0)
                edge_cells.setdefault(key, []).append((k, out_n))
    cuts = unary_union([g.buffer(0.4) for g in cut_lines]) if cut_lines else None
    EPS = 0.015
    def hset(k, x, z, y):
        key = (round(x, 2), round(z, 2))
        hcache[key] = [(kk, yy) for kk, yy in hcache.get(key, ()) if kk != k] + [(k, y)]
    # pre-pass: where two parts' surfaces cross within 0.1 ft of an edge end, the crossing is not inserted as a vertex,
    # so both parts must take ONE height at that end; otherwise a steep plane leaves a sliver seam (review s2-native,
    # Graceland). The lower value wins (the step disappears at that column); the displacement is reported.
    snap_moves = []
    for (a, b), sides in edge_cells.items():
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-6 or len(sides) < 2: continue
        k1, k2 = sides[0][0], sides[1][0]
        if k1 == k2: continue
        d0 = hfun(k1, *a) - hfun(k2, *a); d1 = hfun(k1, *b) - hfun(k2, *b)
        if (d0 > EPS and d1 < -EPS) or (d0 < -EPS and d1 > EPS):
            t = d0 / (d0 - d1)
            end = a if t * L < 0.1 else b if (1 - t) * L < 0.1 else None
            if end is not None:
                y1, y2 = hfun(k1, *end), hfun(k2, *end); y = min(y1, y2)
                hset(k1, *end, y); hset(k2, *end, y); snap_moves.append(round(abs(y1 - y2), 3))
    quads, crossings, base = [], {}, []
    for (a, b), sides in edge_cells.items():
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-6: continue
        ks = {k for k, _ in sides}
        if len(sides) >= 2 and len(ks) == 1: continue           # inside one part (ridge, hip, or a merged sliver)
        is_cut = cuts is not None and cuts.contains(LineString([a, b]))
        pairs = []
        if len(sides) == 1:
            pairs.append((sides[0][0], None, sides[0][1]))
        else:
            (k1, n1), (k2, n2) = sides[0], sides[1]
            pairs += [(k1, k2, n1), (k2, k1, n2)]
        for k, j, (onx, onz) in pairs:
            top0, top1 = hfun(k, *a), hfun(k, *b)
            if j is None: bot0, bot1 = round(float(terr(*a)) - 1.5, 2), round(float(terr(*b)) - 1.5, 2)
            else: bot0, bot1 = hfun(j, *a), hfun(j, *b)
            d0, d1 = top0 - bot0, top1 - bot1
            if max(d0, d1) <= EPS: continue
            if min(d0, d1) >= -EPS: spans = [(0.0, 1.0)]
            else:
                t = d0 / (d0 - d1)
                if t * L < 0.1: spans = [(0.0, 1.0)] if d1 > 0 else []
                elif (1 - t) * L < 0.1: spans = [(0.0, 1.0)] if d0 > 0 else []
                else:
                    spans = [(0.0, t)] if d0 > 0 else [(t, 1.0)]
                    crossings.setdefault((a, b), []).append(t)
                    # the crossing is exported on the 0.01 ft grid; on a steep plane that rounding alone changes the
                    # height by several hundredths, so both parts take ONE height there (the mean of the two)
                    qx, qz = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                    yk, yj = hfun(k, qx, qz), hfun(j, qx, qz)
                    if yk != yj:
                        ym = round((yk + yj) / 2, 2); hset(k, qx, qz, ym); hset(j, qx, qz, ym)
                        snap_moves.append(round(abs(yk - yj) / 2, 3))
            for t0, t1 in spans:
                q0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                q1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                y0t, y1t = hfun(k, *q0), hfun(k, *q1)
                if j is None: y0b, y1b = round(float(terr(*q0)) - 1.5, 2), round(float(terr(*q1)) - 1.5, 2)
                else: y0b, y1b = hfun(j, *q0), hfun(j, *q1)
                if t0 > 0 or y0t - y0b <= EPS: y0b = y0t
                if t1 < 1 or y1t - y1b <= EPS: y1b = y1t
                if y0t < y0b: y0b = y0t
                if y1t < y1b: y1b = y1t
                ex, ez = q1[0] - q0[0], q1[1] - q0[1]
                s0 = 0.0; s1 = math.hypot(ex, ez)
                tgt = 'section' if (is_cut and j is None) else 'walls'
                quads.append((tgt, (q0[0], y0b, q0[1]), (q1[0], y1b, q1[1]), (q1[0], y1t, q1[1]), (q0[0], y0t, q0[1]),
                              s0, s1, (-ez * onx + ex * onz) > 0, j is None))
                if j is None: base.append([[round(q0[0], 2), y0b, round(q0[1], 2)], [round(q1[0], 2), y1b, round(q1[1], 2)]])
    # crossing points become vertices of the roof cells on both sides of their edge (exactly the same point)
    xpts = {}
    for (a, b), ts in crossings.items():
        for t in ts: xpts.setdefault((a, b), []).append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    def ring_with_x(ring):
        out = []
        for a, b in zip(ring[:-1], ring[1:]):
            out.append(a)
            pts = xpts.get((a, b)) or [p for p in reversed(xpts.get((b, a), []))]
            if pts:
                d = (b[0] - a[0], b[1] - a[1]); L2 = d[0] ** 2 + d[1] ** 2
                out += sorted(pts, key=lambda p: ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / L2)
        return out
    roof, roof_flat, walls, section = Buf(), Buf(), Buf(), Buf()
    for c, k, pi in cell_info:
        fit = parts[k]['fit']
        flat = fit.kind in ('flat', 'envelope') or (pi is not None and math.hypot(*fit.planes[pi][:2]) < 0.08)
        buf = roof_flat if flat else roof
        ext = ring_with_x(list(c.exterior.coords)); holes = [ring_with_x(list(h.coords)) for h in c.interiors]
        cp = Polygon(ext, holes)
        if fit.smooth is not None:
            x0, z0, x1, z1 = cp.bounds
            gx, gz = np.meshgrid(np.arange(x0, x1, 3.0), np.arange(z0, z1, 3.0))
            ins = shapely.contains_xy(cp.buffer(-0.5), gx, gz)
            pts = list(cp.exterior.coords)[:-1] + list(zip(gx[ins], gz[ins]))
            tris = [t for t in shapely.delaunay_triangles(shapely.MultiPoint(pts)).geoms if cp.buffer(0.01).contains(t.centroid)]
        else:
            tris = list(shapely.constrained_delaunay_triangles(cp).geoms)
        for t in tris:
            c3 = list(t.exterior.coords)[:3]
            v = [(x, hfun(k, x, z), z) for x, z in c3]
            ax, az = v[1][0] - v[0][0], v[1][2] - v[0][2]; bx, bz = v[2][0] - v[0][0], v[2][2] - v[0][2]
            if az * bx - ax * bz < 0: v = [v[0], v[2], v[1]]; c3 = [c3[0], c3[2], c3[1]]
            buf.tri(*v, *c3)
    # walls as strips between their two vertical sides, carrying every height used at each corner column
    col = {}
    ck = lambda x, z: (round(x, 3), round(z, 3))
    for (_, A, B_, C_, D, *_r) in quads:
        for v in (A, B_, C_, D): col.setdefault(ck(v[0], v[2]), set()).add(v[1])
    for bf in (roof, roof_flat):
        for v in np.asarray(bf.p).reshape(-1, 3):
            key = ck(v[0], v[2])
            if key in col: col[key].add(float(v[1]))
    for (tgt, A, B_, C_, D, s0, s1, keep, ground) in quads:
        buf = section if tgt == 'section' else walls
        def chain(lo, hi):
            ys = sorted(y for y in col.get(ck(lo[0], lo[2]), ()) if lo[1] + 1e-3 < y < hi[1] - 1e-3)
            return [lo] + [(lo[0], y, lo[2]) for y in ys] + [hi]
        Lc, Rc = chain(A, D), chain(B_, C_)
        i = j_ = 0
        while i < len(Lc) - 1 or j_ < len(Rc) - 1:
            if j_ < len(Rc) - 1 and (i == len(Lc) - 1 or Rc[j_ + 1][1] <= Lc[i + 1][1]):
                t = (Lc[i], Rc[j_], Rc[j_ + 1]); tu = ((s0, Lc[i][1]), (s1, Rc[j_][1]), (s1, Rc[j_ + 1][1])); j_ += 1
            else:
                t = (Lc[i], Rc[j_], Lc[i + 1]); tu = ((s0, Lc[i][1]), (s1, Rc[j_][1]), (s0, Lc[i + 1][1])); i += 1
            if keep: buf.tri(t[0], t[1], t[2], *tu)
            else: buf.tri(t[0], t[2], t[1], tu[0], tu[2], tu[1])
    mesh = dict(roof=roof.pack(), roof_flat=roof_flat.pack(), walls=walls.pack(), section=section.pack(), base=base)
    raw = dict(roof=roof.p, roof_flat=roof_flat.p, walls=walls.p, section=section.p)
    mesh_parts.owner = owner
    mesh_parts.snap_moves = snap_moves
    return mesh, raw, hfun, part_index, other


def openings(pg, R, terr, doors, use):
    """Windows on the final exposed walls of every part: equal bays per straight edge, floors from a floor height
    for the use class, kept inside each wall band (above the lower neighbour's roof or the ground, below this
    part's roof), clear of corners and doors. Section cuts and internal faces get none."""
    resid = use in ('house', 'detached', 'residential', 'apartments', 'dormitory', 'semidetached_house', 'terrace', 'townhouse')
    fh = 10.5 if resid else 13.0; bay = 9.5 if resid else 11.0; ww, wh = (3.2, 5.0) if resid else (4.4, 6.4)
    ring = np.asarray(pg.exterior.coords)[:-1]
    base = float(np.percentile([float(terr(x, z)) for x, z in ring], 70))
    wins = []
    for k, prt in enumerate(R['parts']):
      for piece in polys(prt['poly']):
        simp = piece.simplify(0.5, preserve_topology=True)
        co = list(simp.exterior.coords)
        for (x0, z0), (x1, z1) in zip(co[:-1], co[1:]):
            L = math.hypot(x1 - x0, z1 - z0)
            if L < 7: continue
            tx, tz = (x1 - x0) / L, (z1 - z0) / L
            nx, nz = tz, -tx
            mx_, mz_ = (x0 + x1) / 2, (z0 + z1) / 2
            # outward = the side NOT owned by this part (same exact probes as the walls)
            if R['owner'](mx_ + nx * 0.05, mz_ + nz * 0.05) == k: nx, nz = -nx, -nz
            if R['owner'](mx_ - nx * 0.05, mz_ - nz * 0.05) != k: continue      # simplified edge drifted off the part
            ry = math.atan2(nx, nz)
            m = max(1, int((L - 4) // bay)); span = L - 4
            for bix in range(m):
                t = 2 + (bix + 0.5) * span / m
                px, pz = x0 + tx * t, z0 + tz * t
                j = R['other'](px + nx * 0.05, pz + nz * 0.05, k)
                tops = [R['hfun'](k, px + tx * dd, pz + tz * dd) for dd in (-ww / 2, 0, ww / 2)]
                top = min(tops) - 1.2
                if j is None:
                    if not pg.exterior.buffer(0.6).contains(Point(px, pz)) and not any(h.buffer(0.6).contains(Point(px, pz)) for h in pg.interiors):
                        continue                                    # a cut edge, not the building's exterior
                    bot = float(terr(px + nx * 1.5, pz + nz * 1.5)) + 1.2
                else:
                    bot = max(R['hfun'](j, px + tx * dd, pz + tz * dd) for dd in (-ww / 2, 0, ww / 2)) + 1.5
                for f in range(-3, 80):
                    sill = base + f * fh + (2.6 if resid else 3.0)
                    if sill + wh > top: break
                    if sill < bot: continue
                    if j is None and any(math.hypot(px - d[0], pz - d[2]) < d[4] / 2 + ww / 2 + 1.5 and sill < d[1] + 9 for d in doors): continue
                    wins.append([round(px + nx * 0.06, 2), round(sill + wh / 2, 2), round(pz + nz * 0.06, 2), round(ry, 4), ww, wh])
    return wins, fh, base


# ------------------------------------------------------------------ S2.3 planar B-rep fallback
def brep_split(pp, T, pg, cell=None):
    """Support-constrained piecewise-planar roof for a part no primitive explains (review R2, F1).
    1. Robust height raster of the part's own training returns; region-growing plane segmentation; coplanar merge;
       planes with too little support are absorbed by neighbours.
    2. One polygon per plane region, simplified together as a coverage (shared boundaries identical), clipped to the
       part.
    3. Joints: a boundary vertex shared by exactly two regions moves onto the line where their planes intersect,
       when that line passes within reach and the planes agree there (a ridge or valley); a vertex shared by three
       regions moves to the least-squares meeting point of their three lines. Elsewhere the boundary is a step
       (a wall). Only adjacent supported planes are ever intersected.
    Returns [(polygon, plane (a, b, c), n_points)] or None."""
    import analyze_site as A
    from scipy import ndimage
    pts = _pts_in(pp, T)
    if len(pts) < 40: return None
    if cell is None: cell = max(1.5, min(3.0, 1.2 / math.sqrt(max(len(pts) / pp.area, 1e-3))))
    P4 = np.c_[pts, np.ones(len(pts))]
    G, H = height_raster(pp, P4, cell)
    have = G['inside'] & ~np.isnan(H)
    if have.sum() < 12: return None
    idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
    Hf = np.where(G['inside'], ndimage.median_filter(H[tuple(idx)], size=3), np.nan)
    label, planes = A.segment_heights(np.nan_to_num(Hf), G['inside'], G['GX'], G['GZ'])
    if (label == 0).any() and (label > 0).any():
        idx = ndimage.distance_transform_edt(label == 0, return_distances=False, return_indices=True)
        label = np.where(G['inside'], label[tuple(idx)], 0)
    L0 = np.where(G['inside'], label - 1, -1)
    if L0.max() < 1: return None                        # one plane: the primitives already covered it
    lab, planes = A.merge_coplanar(np.where(L0 < 0, 0, L0), planes, np.nan_to_num(Hf), G['inside'], G['GX'], G['GZ'])
    lab = np.where(G['inside'], lab, -1)
    lab = np.where(G['inside'], A.absorb_fragments(np.where(lab < 0, 0, lab), min_cells=int(math.ceil(60 / cell ** 2))), -1)
    ks = [k for k in np.unique(lab) if k >= 0]
    if len(ks) < 2 or len(ks) > 24: return None
    # region polygons from cells, simplified as one coverage
    c = G['cell']; cen, ux, vx = G['cen'], G['ux'], G['vx']
    def cellpoly(i, j):
        u0 = G['u0'] + j * c; v0 = G['v0'] + i * c
        q = [(u0, v0), (u0 + c, v0), (u0 + c, v0 + c), (u0, v0 + c)]
        return Polygon([(cen[0] + a * ux[0] + b * vx[0], cen[1] + a * ux[1] + b * vx[1]) for a, b in q])
    regs = [unary_union([cellpoly(i, j) for i, j in zip(*np.nonzero(lab == k))]).buffer(0) for k in ks]
    regs = list(shapely.coverage_simplify(np.array(regs, dtype=object), tolerance=max(1.0, c * 0.8), simplify_boundary=False))
    regs = [max(polys(r.intersection(pp)), key=lambda q: q.area) if polys(r.intersection(pp)) else None for r in regs]
    keep = [(r, k) for r, k in zip(regs, ks) if r is not None and r.area > 20]
    if len(keep) < 2: return None
    regs = [r for r, _ in keep]; PL = [tuple(planes[k]) for _, k in keep]
    # vertex -> regions using it
    use = {}
    for ri, r in enumerate(regs):
        for x, z in list(r.exterior.coords)[:-1]: use.setdefault((round(x, 6), round(z, 6)), set()).add(ri)
    boundary = pp.boundary
    reach = max(2.0, 2 * c)
    moved = {}
    def line(i, j):
        (a1, b1, c1), (a2, b2, c2) = PL[i], PL[j]
        return a1 - a2, b1 - b2, c1 - c2                 # (A, B, C): A x + B z + C = 0
    for key, rs in use.items():
        x, z = key
        if boundary.distance(Point(x, z)) < 0.05: continue          # the part outline itself never moves
        rs = sorted(rs)
        if len(rs) == 2:
            A_, B_, C_ = line(*rs); nrm = math.hypot(A_, B_)
            if nrm < 1e-6: continue
            d = (A_ * x + B_ * z + C_) / nrm
            if abs(d) <= reach:
                nx_, nz_ = x - A_ / nrm * d, z - B_ / nrm * d
                moved[key] = (nx_, nz_)
        elif len(rs) >= 3:
            rows = [line(rs[0], rs[1]), line(rs[1], rs[2]), line(rs[0], rs[2])]
            M = np.array([[A_, B_] for A_, B_, _ in rows]); rhs = -np.array([C_ for _, _, C_ in rows])
            if np.linalg.matrix_rank(M) < 2: continue
            sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            if math.hypot(sol[0] - x, sol[1] - z) <= reach: moved[key] = (float(sol[0]), float(sol[1]))
    def mv(r):
        ext = [moved.get((round(x, 6), round(z, 6)), (x, z)) for x, z in list(r.exterior.coords)[:-1]]
        q = Polygon(ext).buffer(0)
        return q
    regs2 = [mv(r) for r in regs]
    # keep the result a coverage of the part: clip, then give any uncovered area to the most similar neighbour
    regs2 = [max(polys(r.intersection(pp)), key=lambda q: q.area) if polys(r.intersection(pp)) else None for r in regs2]
    if any(r is None for r in regs2): return None
    for i in range(len(regs2)):
        for j in range(i):
            regs2[i] = regs2[i].difference(regs2[j])
    regs2 = [max(polys(r), key=lambda q: q.area) if polys(r) else None for r in regs2]
    if any(r is None or r.area < 10 for r in regs2): return None
    rest = pp.difference(unary_union(regs2))
    for r in polys(rest):
        if r.area < 0.5: continue
        k = max(range(len(regs2)), key=lambda t: regs2[t].intersection(r.buffer(0.05)).area)
        regs2[k] = unary_union([regs2[k], r]).buffer(0)
    out = []
    for r, pl in zip(regs2, PL):
        r = max(polys(r), key=lambda q: q.area)
        n = int(shapely.contains_xy(r, pts[:, 0], pts[:, 2]).sum())
        out.append((r, pl, n))
    return out
