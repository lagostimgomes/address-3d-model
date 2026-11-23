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
    idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
    Hf = ndimage.median_filter(H[tuple(idx)], size=3)
    # connect 4-neighbours whose heights differ by less than STEP_FT, inside the footprint
    R, C = Hf.shape
    lab = -np.ones((R, C), int); n = 0
    from collections import deque
    for i0 in range(R):
        for j0 in range(C):
            if not ins[i0, j0] or lab[i0, j0] >= 0: continue
            q = deque([(i0, j0)]); lab[i0, j0] = n
            while q:
                i, j = q.popleft()
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < R and 0 <= b < C and ins[a, b] and lab[a, b] < 0 and abs(Hf[a, b] - Hf[i, j]) < STEP_FT:
                        lab[a, b] = n; q.append((a, b))
            n += 1
    # small regions join the neighbour with the most similar height
    cell2 = G['cell'] ** 2
    for _ in range(4):
        sizes = np.bincount(lab[lab >= 0], minlength=n)
        meds = {k: float(np.median(Hf[lab == k])) for k in range(n) if sizes[k]}
        changed = False
        for k in sorted(range(n), key=lambda k: sizes[k]):
            if not sizes[k] or sizes[k] * cell2 >= MIN_PART_FT2: continue
            m = lab == k; ring = ndimage.binary_dilation(m) & ~m & (lab >= 0)
            nb = np.unique(lab[ring]); nb = nb[nb != k]
            if not len(nb): continue
            tgt = min(nb, key=lambda t: abs(meds.get(t, 1e9) - meds[k]))
            lab[m] = tgt; changed = True
            sizes = np.bincount(lab[lab >= 0], minlength=n)
        if not changed: break
    ks = [k for k in np.unique(lab) if k >= 0]
    if len(ks) == 1:
        return [pg], [float(np.median(Hf[ins]))]
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
                parts.append(p); meds.append(float(np.median(Hf[lab == k])))
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
    return [shapely.set_precision(p, GRID) for p in out]


def _pts_in(pp, T):
    inn = shapely.contains_xy(pp.buffer(0.5), T[:, 0], T[:, 2]) if len(T) else np.zeros(0, bool)
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
    G, H = height_raster(pg, T, cell)
    part_polys, meds = partition(pg, G, H)
    # parts that no single primitive explains are split further by the best straight cut (recursive)
    refined = []
    for pp in part_polys: refined += refine_part(pp, T, pg=pg)
    part_polys = clean_partition(refined, pg)
    parts = []
    for k, pp in enumerate(part_polys):
        inn = shapely.contains_xy(pp.buffer(0.5), T[:, 0], T[:, 2]) if len(T) else np.zeros(0, bool)
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
        cov = float(np.mean(~np.isnan(H[shapely.contains_xy(pp, G['GX'], G['GZ'])]))) if shapely.contains_xy(pp, G['GX'], G['GZ']).any() else None
        st.update(heldout=dict(n=int(len(vr)), median_abs=round(float(np.median(vr)), 2) if len(vr) else None,
                               p90_abs=round(float(np.percentile(vr, 90)), 2) if len(vr) else None, units='ft',
                               underpowered=bool(len(vr) < 50)),
                  coverage=None if cov is None else round(cov, 3), density=round(len(pts) / max(pp.area, 1), 3), area_ft2=round(pp.area))
        parts.append(dict(poly=pp, fit=fit, stats=st))
    mesh, raw, hfun, part_index, other = mesh_parts(parts, terr, cut_lines)
    owner = mesh_parts.owner
    return dict(parts=parts, mesh=mesh, raw=raw, hfun=hfun, part_index=part_index, other=other, owner=owner, holdout_n=int(test.sum()))


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
    cells = [c for c in polygonize(lines) if c.area > 1e-4]
    part_index = shapely.STRtree([p_['poly'] for p_ in parts])
    def owner(x, z):
        pt = Point(x, z)
        for j in part_index.query(pt):
            if parts[j]['poly'].contains(pt): return int(j)
        return None
    hcache = {}
    def hfun(k, x, z):
        key = (round(x, 3), round(z, 3))
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
    # from here on the arrangement is the only geometry: each part is the union of its cells
    for k in range(len(parts)):
        cs = [c for c, kk, _ in cell_info if kk == k]
        if cs: parts[k]['poly'] = unary_union(cs)
    part_index = shapely.STRtree([p_['poly'] for p_ in parts])
    # arrangement edges with the part on each side; walls where the two sides differ in height
    segs = []
    for ln in getattr(lines, 'geoms', [lines]):
        co = list(ln.coords)
        segs += list(zip(co[:-1], co[1:]))
    cuts = unary_union([g.buffer(0.4) for g in cut_lines]) if cut_lines else None
    EPS_SIDE, EPS = 0.02, 0.015
    quads, crossings, base = [], {}, []
    for a, b in segs:
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-6: continue
        nx, nz = (b[1] - a[1]) / L, -(b[0] - a[0]) / L
        mx, mz = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        kp, km = owner(mx + nx * EPS_SIDE, mz + nz * EPS_SIDE), owner(mx - nx * EPS_SIDE, mz - nz * EPS_SIDE)
        if kp == km: continue                                     # inside one part (a ridge or hip line): no wall
        is_cut = cuts is not None and cuts.contains(LineString([a, b]))
        for k, j, onx, onz in ((km, kp, nx, nz), (kp, km, -nx, -nz)):
            if k is None: continue                                # the wall is drawn from the higher side only
            top0, top1 = hfun(k, *a), hfun(k, *b)
            if j is None: bot0, bot1 = round(float(terr(*a)) - 1.5, 2), round(float(terr(*b)) - 1.5, 2)
            else: bot0, bot1 = hfun(j, *a), hfun(j, *b)
            d0, d1 = top0 - bot0, top1 - bot1
            if max(d0, d1) <= EPS: continue
            if min(d0, d1) >= -EPS: spans = [(0.0, 1.0)]
            else:
                t = d0 / (d0 - d1)
                if t * L < 0.1:                        # crossing at an endpoint: snap to it (no sliver vertex)
                    spans = [(0.0, 1.0)] if d1 > 0 else []
                elif (1 - t) * L < 0.1:
                    spans = [(0.0, 1.0)] if d0 > 0 else []
                else:
                    spans = [(0.0, t)] if d0 > 0 else [(t, 1.0)]
                    crossings.setdefault((a, b), []).append(t)
            for t0, t1 in spans:
                q0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                q1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                y0t, y1t = hfun(k, *q0), hfun(k, *q1)
                if j is None: y0b, y1b = round(float(terr(*q0)) - 1.5, 2), round(float(terr(*q1)) - 1.5, 2)
                else: y0b, y1b = hfun(j, *q0), hfun(j, *q1)
                if t0 > 0 or y0t - y0b <= EPS: y0b = y0t
                if t1 < 1 or y1t - y1b <= EPS: y1b = y1t
                if y0t < y0b: y0b = y0t                  # snapped crossing at an end: that end has zero height
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
