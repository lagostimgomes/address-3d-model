"""
geometry.py - turn a building footprint (polygon) plus measured roof planes into explicit meshes,
and lay out entrances and windows wall by wall.

Model frame: feet, +x east, +z south; y is height relative to PAD. Roof planes live in a frame
rotated by `ang` (u = x cos + z sin, v = -x sin + z cos), y = a*u + b*v + c.

Provenance is kept apart from decoration: outlines are 'mapped' (footprint data) or 'measured'
(LiDAR), heights and roof planes are 'measured', windows are always 'generated', and doors are
'mapped' (an entrance node), 'inferred-path' (a mapped footpath ends at the wall) or
'inferred-street' (the wall faces the nearest street).
"""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box as sbox
from shapely.prepared import prep

FLAT = 0.08          # |grad| below this is a flat roof (about 4.6 deg)
PARAPET = 2.6        # ft above a flat roof at its outer edge
STEP_MIN = 0.25      # ft: smaller steps between planes are not walled


class Planes:
    """Label raster + plane list in the rotated (u, v) frame."""

    def __init__(self, label, planes, ang, u0, v0, cell, hgt=None):
        self.L, self.P = label, np.asarray(planes, float)
        self.H = hgt
        self.ang, self.u0, self.v0, self.c = ang, u0, v0, cell
        self.ca, self.sa = math.cos(ang), math.sin(ang)
        self.R, self.C = label.shape
        self._corners()

    def _corners(self):
        """Corner heights of every cell on its own plane. CH[i, j, k]: k = 0 (v0,u0), 1 (v1,u0), 2 (v1,u1), 3 (v0,u1)."""
        R, C, c, P = self.R, self.C, self.c, self.P
        I, J = np.meshgrid(np.arange(R), np.arange(C), indexing='ij')
        A, Bc, Cc = P[self.L][..., 0], P[self.L][..., 1], P[self.L][..., 2]
        CH = np.zeros((R, C, 4))
        for k, (di, dj) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            CH[..., k] = A * (self.u0 + (J + dj) * c) + Bc * (self.v0 + (I + di) * c) + Cc
        self.CH = CH

    def cell_of(self, u, v):
        return (int(np.clip(math.floor((v - self.v0) / self.c), 0, self.R - 1)),
                int(np.clip(math.floor((u - self.u0) / self.c), 0, self.C - 1)))

    def yc(self, i, j, u, v):
        """Roof height at (u, v) on cell (i, j)'s surface (bilinear in its corners)."""
        s_ = (u - (self.u0 + j * self.c)) / self.c; t_ = (v - (self.v0 + i * self.c)) / self.c
        h = self.CH[i, j]
        return (h[0] * (1 - s_) * (1 - t_) + h[1] * (1 - s_) * t_ + h[2] * s_ * t_ + h[3] * s_ * (1 - t_))

    def y_at(self, x, z, nx=0.0, nz=0.0):
        """Roof height at a point of the outline; (nx, nz) points into the building, to pick the cell inside."""
        u, v = self.uv(x + nx * 0.35, z + nz * 0.35); i, j = self.cell_of(u, v)
        u, v = self.uv(x, z)
        return self.yc(i, j, u, v), self.flat(self.L[i, j])

    def uv(self, x, z):
        return x * self.ca + z * self.sa, -x * self.sa + z * self.ca

    def xz(self, u, v):
        return u * self.ca - v * self.sa, u * self.sa + v * self.ca

    def lab_at_uv(self, u, v):
        j = int(np.clip(math.floor((u - self.u0) / self.c), 0, self.C - 1))
        i = int(np.clip(math.floor((v - self.v0) / self.c), 0, self.R - 1))
        return int(self.L[i, j])

    def y(self, l, u, v):
        a, b, c = self.P[l]
        return a * u + b * v + c

    flat_thr = FLAT

    def flat(self, l):
        return math.hypot(self.P[l][0], self.P[l][1]) < self.flat_thr


def as_polys(g):
    if g is None or g.is_empty: return []
    if isinstance(g, Polygon): return [g]
    if isinstance(g, MultiPolygon): return list(g.geoms)
    return [p for p in getattr(g, 'geoms', []) if isinstance(p, Polygon)]


def tris_of(poly):
    """Triangles (list of 3 (u, v) tuples) covering a polygon exactly, holes respected."""
    out = []
    for p in as_polys(poly):
        if p.area < 1e-4: continue
        coords = list(p.exterior.coords)
        if len(coords) == 4 and not p.interiors:
            out.append(coords[:3]); continue
        if len(coords) == 5 and not p.interiors and abs(p.area - p.minimum_rotated_rectangle.area) < 1e-6:
            a, b, c, d = coords[:4]; out += [(a, b, c), (a, c, d)]; continue
        tri = shapely.constrained_delaunay_triangles(p)
        for t in tri.geoms:
            out.append(list(t.exterior.coords)[:3])
    return out


class MeshBuf:
    def __init__(self):
        self.p, self.uv, self.nrm = [], [], []

    def tri(self, a, b, c, ua=None, ub=None, uc=None):
        self.p += [*a, *b, *c]
        if ua is not None: self.uv += [*ua, *ub, *uc]

    def quad(self, a, b, c, d, ua=None, ub=None, uc=None, ud=None):
        """a-b bottom edge, c above b, d above a (counter-clockwise seen from the front)."""
        self.tri(a, b, c, ua, ub, uc); self.tri(a, c, d, ua, uc, ud)

    def pack(self):
        """Int16 at a 0.05 ft quantum (range +-1,638 ft), base64: about a third of the size of JSON numbers."""
        import base64
        q = lambda v: base64.b64encode(np.clip(np.round(np.asarray(v) * 20), -32767, 32767).astype('<i2').tobytes()).decode()
        out = dict(p=q(self.p), uv=q(self.uv) if self.uv else None, n=len(self.p))
        if self.nrm and len(self.nrm) == len(self.p):
            out['nrm'] = base64.b64encode(np.round(np.asarray(self.nrm) * 127).astype('i1').tobytes()).decode()
        return out


def roof_and_walls(poly, pl, terr, cut_lines=(), parapet=True):
    """Meshes for one building: roof (pitched / flat), outer walls, step walls, fascia and section faces.
    poly: footprint in model (x, z). pl: Planes. terr(x, z) -> ground y. cut_lines: geometries along which the
    outline was cut (site edge or data edge): walls there get the section material."""
    roof, rflat, walls, steps, fascia, section = MeshBuf(), MeshBuf(), MeshBuf(), MeshBuf(), MeshBuf(), MeshBuf()
    puv = shapely.transform(poly, lambda c: np.c_[c[:, 0] * pl.ca + c[:, 1] * pl.sa, -c[:, 0] * pl.sa + c[:, 1] * pl.ca])
    pp = prep(puv)
    c = pl.c
    X = lambda u, v, y: (round(u * pl.ca - v * pl.sa, 3), round(y, 3), round(u * pl.sa + v * pl.ca, 3))

    # ---- roof: each cell on its plane; runs of whole cells on one plane merge into one quad
    gu = pl.u0 + (np.arange(pl.C) + .5) * c; gv = pl.v0 + (np.arange(pl.R) + .5) * c
    GU, GV = np.meshgrid(gu, gv)
    inside = shapely.contains_xy(puv.buffer(-c * 0.72), GU, GV)
    near = shapely.contains_xy(puv.buffer(c * 0.72), GU, GV) & ~inside
    touch = np.zeros_like(inside)
    for i, j in zip(*np.nonzero(near)):
        if pp.intersects(sbox(pl.u0 + j * c, pl.v0 + i * c, pl.u0 + (j + 1) * c, pl.v0 + (i + 1) * c)): touch[i, j] = True
    for i in range(pl.R):
        v0, v1 = pl.v0 + i * c, pl.v0 + (i + 1) * c
        j = 0
        while j < pl.C:
            if inside[i, j]:
                l = pl.L[i, j]; k = j + 1
                while k < pl.C and inside[i, k] and pl.L[i, k] == l: k += 1
                u0, u1 = pl.u0 + j * c, pl.u0 + k * c
                a_, b_, cc, d = (u0, v0), (u0, v1), (u1, v1), (u1, v0)
                (rflat if pl.flat(l) else roof).quad(*(X(u, v, pl.y(l, u, v)) for u, v in (a_, b_, cc, d)), a_, b_, cc, d)
                j = k; continue
            if touch[i, j]:
                l = pl.L[i, j]
                for t in tris_of(puv.intersection(sbox(pl.u0 + j * c, v0, pl.u0 + (j + 1) * c, v1))):
                    pts = [X(u, v, pl.y(l, u, v)) for u, v in t]
                    ax, az = pts[1][0] - pts[0][0], pts[1][2] - pts[0][2]; bx, bz = pts[2][0] - pts[0][0], pts[2][2] - pts[0][2]
                    if az * bx - ax * bz < 0: pts = [pts[0], pts[2], pts[1]]; t = [t[0], t[2], t[1]]
                    (rflat if pl.flat(l) else roof).tri(*pts, *t)
            j += 1

    # ---- step walls where neighbouring cells sit on different planes (vertical, along cell edges)
    use = inside | touch
    def wall_edge(la, lb, p0, p1, clip):
        if clip:
            seg = LineString([p0, p1]).intersection(puv)
            parts = [g for g in getattr(seg, 'geoms', [seg]) if not g.is_empty and g.geom_type == 'LineString' and g.length > 0.03]
            ends = [(g.coords[0], g.coords[-1]) for g in parts]
        else:
            ends = [(p0, p1)]
        for (au, av), (bu, bv) in ends:
            ya0, ya1, yb0, yb1 = pl.y(la, au, av), pl.y(la, bu, bv), pl.y(lb, au, av), pl.y(lb, bu, bv)
            if max(abs(ya0 - yb0), abs(ya1 - yb1)) < STEP_MIN: continue
            lo0, hi0, lo1, hi1 = min(ya0, yb0), max(ya0, yb0), min(ya1, yb1), max(ya1, yb1)
            L = math.hypot(bu - au, bv - av)
            if max(hi0 - lo0, hi1 - lo1) < 1.5:     # a small ledge reads as roof, not as a strip of wall
                (rflat if pl.flat(la) and pl.flat(lb) else roof).quad(X(au, av, lo0), X(bu, bv, lo1), X(bu, bv, hi1), X(au, av, hi0),
                                                                    (au, av), (bu, bv), (bu, bv), (au, av))
                continue
            # shade the step as one smooth wall: horizontal normal from the smoothed height field, pointing to the low side
            mu, mv = (au + bu) / 2, (av + bv) / 2
            gi = (mv - pl.v0) / c - .5; gj = (mu - pl.u0) / c - .5
            gy = ndimage.map_coordinates(dHv, [[gi], [gj]], order=1, mode='nearest')[0]
            gx = ndimage.map_coordinates(dHu, [[gi], [gj]], order=1, mode='nearest')[0]
            nl = math.hypot(gx, gy)
            if nl < 1e-6: nu_, nv_ = (bv - av) / L, -(bu - au) / L
            else: nu_, nv_ = -gx / nl, -gy / nl
            nx_, nz_ = nu_ * pl.ca - nv_ * pl.sa, nu_ * pl.sa + nv_ * pl.ca
            A0, B0 = X(au, av, lo0), X(bu, bv, lo1)
            ta = (-nz_ * A0[0] + nx_ * A0[2]) % 960; tb = ta + (-nz_ * (B0[0] - A0[0]) + nx_ * (B0[2] - A0[2]))     # along the smoothed wall
            steps.quad(A0, B0, X(bu, bv, hi1), X(au, av, hi0), (ta, lo0), (tb, lo1), (tb, hi1), (ta, hi0))
            steps.nrm += [round(nx_, 3), 0, round(nz_, 3)] * 6
    from scipy import ndimage
    Hc = pl.P[pl.L][..., 0] * GU + pl.P[pl.L][..., 1] * GV + pl.P[pl.L][..., 2]
    Hs = ndimage.gaussian_filter(Hc, 2.0)
    dHv, dHu = np.gradient(Hs)
    for i, j in zip(*np.nonzero(use[:, :-1] & use[:, 1:] & (pl.L[:, :-1] != pl.L[:, 1:]))):
        u1, v0 = pl.u0 + (j + 1) * c, pl.v0 + i * c
        wall_edge(pl.L[i, j], pl.L[i, j + 1], (u1, v0), (u1, v0 + c), touch[i, j] or touch[i, j + 1])
    for i, j in zip(*np.nonzero(use[:-1, :] & use[1:, :] & (pl.L[:-1, :] != pl.L[1:, :]))):
        u0, v1 = pl.u0 + j * c, pl.v0 + (i + 1) * c
        wall_edge(pl.L[i, j], pl.L[i + 1, j], (u0, v1), (u0 + c, v1), touch[i, j] or touch[i + 1, j])

    # ---- outer walls along the outline (exterior and courtyards), top on the roof plane at the wall
    cuts = shapely.union_all([g.buffer(0.4) for g in cut_lines]) if cut_lines else None
    profile = []          # per ring: list of (x, z, top, ground, flat?) samples, for facades
    rings = [poly.exterior] + list(poly.interiors)
    for ring in rings:
        co = list(ring.coords)
        samples = []
        for (x0, z0), (x1, z1) in zip(co[:-1], co[1:]):
            L = math.hypot(x1 - x0, z1 - z0)
            if L < 1e-3: continue
            nseg = max(1, int(math.ceil(L / (c * 0.75))))
            # inward normal: the side the polygon is on
            nx, nz = -(z1 - z0) / L, (x1 - x0) / L
            mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
            if not poly.contains(Point(mx + nx * 0.3, mz + nz * 0.3)): nx, nz = -nx, -nz
            is_cut = cuts is not None and cuts.contains(LineString([(x0, z0), (x1, z1)]))
            for s in range(nseg):
                t0, t1 = s / nseg, (s + 1) / nseg
                a = (x0 + (x1 - x0) * t0, z0 + (z1 - z0) * t0); b = (x0 + (x1 - x0) * t1, z0 + (z1 - z0) * t1)
                tops = []
                for (px, pz) in (a, b):
                    y, fl = pl.y_at(px, pz, nx, nz)
                    if parapet and fl and not is_cut: y += PARAPET
                    tops.append((y, fl))
                ga, gb = float(terr(*a)) - 1.5, float(terr(*b)) - 1.5
                buf = section if is_cut else walls
                d0 = (math.hypot(a[0] - co[0][0], a[1] - co[0][1])) % 960; d1 = d0 + L / nseg   # 960 ft: a multiple of every texture period; keeps UVs inside int16
                buf.quad((a[0], ga, a[1]), (b[0], gb, b[1]), (b[0], tops[1][0], b[1]), (a[0], tops[0][0], a[1]),
                         (d0, ga), (d1, gb), (d1, tops[1][0]), (d0, tops[0][0]))
                if not is_cut:
                    o = 0.25; h = 0.9 if not tops[0][1] else 0.55
                    fascia.quad((a[0] - nx * o, tops[0][0] - h, a[1] - nz * o), (b[0] - nx * o, tops[1][0] - h, b[1] - nz * o),
                                (b[0] - nx * o, tops[1][0] + 0.08, b[1] - nz * o), (a[0] - nx * o, tops[0][0] + 0.08, a[1] - nz * o))
                    if tops[0][1] and parapet:   # inner face of the parapet, seen from above
                        ia, ib = (a[0] + nx * 0.7, a[1] + nz * 0.7), (b[0] + nx * 0.7, b[1] + nz * 0.7)
                        ra = pl.y_at(*ia, nx, nz)[0]; rb = pl.y_at(*ib, nx, nz)[0]
                        walls.quad((ib[0], rb, ib[1]), (ia[0], ra, ia[1]), (ia[0], tops[0][0], ia[1]), (ib[0], tops[1][0], ib[1]),
                                   (d1, rb), (d0, ra), (d0, tops[0][0]), (d1, tops[1][0]))
                        rflat.quad((a[0], tops[0][0], a[1]), (ia[0], tops[0][0], ia[1]), (ib[0], tops[1][0], ib[1]), (b[0], tops[1][0], b[1]),
                                   (0, 0), (0, 1), (1, 1), (1, 0))
                samples.append((a, b, tops, (ga + 1.5, gb + 1.5), is_cut, (nx, nz)))
        profile.append(samples)
    return dict(roof=roof.pack(), roof_flat=rflat.pack(), walls=walls.pack(), steps=steps.pack(),
                fascia=fascia.pack(), section=section.pack()), profile


def roof_y(pl, x, z, nx=0.0, nz=0.0):
    return pl.y_at(x, z, nx, nz)


# ---------------------------------------------------------------- entrances
def entrances(poly, ways, nodes, roads_xy, terr, big, drive_xy=None, garage_ok=False):
    """Doors on this building: mapped entrance nodes, then footpaths that end at the wall, else (for larger
    buildings) the middle of the longest wall facing the nearest street. Returns [x, y, z, ry, width, kind]."""
    ext = poly.exterior
    found = []

    def add(px, pz, kind):
        d = ext.project(Point(px, pz)); p = ext.interpolate(d)
        a = ext.interpolate(max(0, d - 0.5)); b = ext.interpolate(min(ext.length, d + 0.5))
        tx, tz = b.x - a.x, b.y - a.y; L = math.hypot(tx, tz) or 1
        nx, nz = tz / L, -tx / L
        if poly.contains(Point(p.x + nx * 0.5, p.y + nz * 0.5)): nx, nz = -nx, -nz
        if any(math.hypot(p.x - f[0], p.y - f[2]) < 14 for f in found): return
        found.append([round(p.x, 2), round(float(terr(p.x + nx * 2, p.y + nz * 2)), 2), round(p.y, 2),
                      round(math.atan2(nx, nz), 4), 6.0 if big else 3.4, kind])
    for e in nodes:
        if ext.distance(Point(e['x'], e['z'])) < 6: add(e['x'], e['z'], 'mapped')
    for w in ways:
        if w['kind'] not in ('footway', 'path', 'steps', 'pedestrian', 'corridor'): continue
        pts = w['centerline']
        for k in (0, len(pts) - 1):
            px, pz = pts[k]
            if ext.distance(Point(px, pz)) < 6:
                q = pts[1] if k == 0 else pts[-2]
                if not poly.contains(Point(q)): add(px, pz, 'inferred-path')
    # every building gets one street door when nothing better is known (C3: a plain inferred door is allowed)
    if not any(f[5] != 'inferred-driveway' for f in found) and roads_xy is not None and len(roads_xy):
        segs = list(zip(ext.coords[:-1], ext.coords[1:]))
        best = None
        for (x0, z0), (x1, z1) in segs:
            L = math.hypot(x1 - x0, z1 - z0)
            if L < 10: continue
            mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
            nx, nz = (z1 - z0) / L, -(x1 - x0) / L
            if poly.contains(Point(mx + nx * 0.5, mz + nz * 0.5)): nx, nz = -nx, -nz
            d = np.hypot(roads_xy[:, 0] - mx, roads_xy[:, 1] - mz); k = int(np.argmin(d))
            vx, vz = roads_xy[k, 0] - mx, roads_xy[k, 1] - mz; vl = math.hypot(vx, vz) or 1
            face = (nx * vx + nz * vz) / vl
            score = face * min(L, 60) / (1 + d[k] / 200)
            if best is None or score > best[0]: best = (score, mx, mz)
        if best and best[0] > 0: add(best[1], best[2], 'inferred-street')
    # garage doors: where estimated paving (a driveway) meets a wall of a house-sized building along >= 9 ft.
    # Evidence: paved cells within 5 ft outside the wall; the door is centred on the paved stretch.
    if garage_ok and drive_xy is not None and len(drive_xy):
        from shapely.geometry import MultiPoint
        simp = poly.simplify(1.0, preserve_topology=True)
        co = list(simp.exterior.coords)
        near = drive_xy[np.hypot(drive_xy[:, 0] - poly.centroid.x, drive_xy[:, 1] - poly.centroid.y) < math.sqrt(poly.area) * 2 + 20]
        for (x0, z0), (x1, z1) in zip(co[:-1], co[1:]):
            L = math.hypot(x1 - x0, z1 - z0)
            if L < 10 or not len(near): continue
            tx, tz = (x1 - x0) / L, (z1 - z0) / L; nx, nz = tz, -tx
            mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
            if poly.contains(Point(mx + nx * 0.5, mz + nz * 0.5)): nx, nz = -nx, -nz
            rel = near - [x0, z0]
            s_ = rel @ [tx, tz]; o_ = rel @ [nx, nz]
            m = (s_ > 1.5) & (s_ < L - 1.5) & (o_ > 0.5) & (o_ < 5.0)
            if m.sum() < 6: continue
            ss = np.sort(s_[m]); gaps = np.diff(ss)
            # the longest run of paved cells with no gap over 3 ft
            runs, st_ = [], 0
            for k in range(len(gaps)):
                if gaps[k] > 3.0: runs.append((ss[st_], ss[k])); st_ = k + 1
            runs.append((ss[st_], ss[-1]))
            a_, b_ = max(runs, key=lambda r: r[1] - r[0])
            if b_ - a_ < 9.0: continue
            w_ = 16.0 if b_ - a_ >= 18 else 8.6
            c_ = (a_ + b_) / 2
            px, pz = x0 + tx * c_, z0 + tz * c_
            if any(math.hypot(px - f[0], pz - f[2]) < 10 for f in found): continue
            found.append([round(px, 2), round(float(terr(px + nx * 2, pz + nz * 2)), 2), round(pz, 2),
                          round(math.atan2(nx, nz), 4), w_, 'inferred-driveway'])
    return found


# ---------------------------------------------------------------- windows
def facade(poly, pl, terr, profile, doors, use):
    """Windows laid out wall by wall: equal bays per straight wall, floors from a floor height that suits the
    building's use, nothing within 2 ft of a corner, above the roof line, below ground, or across a door.
    Returns [[x, y_center, z, ry, w, h], ...] and the floor height used."""
    resid = use in ('house', 'detached', 'residential', 'apartments', 'dormitory', 'semidetached_house', 'terrace', 'townhouse')
    fh = 10.5 if resid else 13.0
    bay = 9.5 if resid else 11.0
    ww, wh = (3.2, 5.0) if resid else (4.4, 6.4)
    simple = poly.simplify(1.0, preserve_topology=True)
    ring_xy = list(simple.exterior.coords)
    gr = [float(terr(x, z)) for x, z in ring_xy[:-1]]
    base = float(np.percentile(gr, 70))          # main floor level: on slopes, the upper side's grade
    wins = []
    for (x0, z0), (x1, z1) in zip(ring_xy[:-1], ring_xy[1:]):
        L = math.hypot(x1 - x0, z1 - z0)
        if L < 7: continue
        tx, tz = (x1 - x0) / L, (z1 - z0) / L
        nx, nz = tz, -tx
        if poly.contains(Point((x0 + x1) / 2 + nx * 0.5, (z0 + z1) / 2 + nz * 0.5)): nx, nz = -nx, -nz
        if not poly.exterior.buffer(0.6).contains(LineString([(x0, z0), (x1, z1)])): continue   # cut edges: no windows
        ry = math.atan2(nx, nz)
        m = max(1, int((L - 4) // bay))
        span = L - 4
        for b in range(m):
            t = 2 + (b + 0.5) * span / m
            px, pz = x0 + tx * t, z0 + tz * t
            g = float(terr(px + nx * 1.5, pz + nz * 1.5))
            top_here = min(roof_y(pl, px + tx * dd, pz + tz * dd, -nx, -nz)[0] for dd in (-ww / 2, 0, ww / 2))
            flat_here = roof_y(pl, px, pz, -nx, -nz)[1]
            lim = top_here - (1.4 if flat_here else 1.0)
            for k in range(-3, 60):
                sill = base + k * fh + (2.6 if resid else 3.0)
                if sill + wh > lim: break
                if sill < g + 1.2: continue
                if any(math.hypot(px - d[0], pz - d[2]) < d[4] / 2 + ww / 2 + 1.5 and sill < d[1] + 9 for d in doors): continue
                wins.append([round(px + nx * 0.06, 2), round(sill + wh / 2, 2), round(pz + nz * 0.06, 2), round(ry, 4), ww, wh])
    return wins, fh, base
