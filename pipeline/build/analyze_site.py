#!/usr/bin/env python3
"""
analyze_site.py - measure the featured house, one neighbor on each side, and trees from
USGS 3DEP LiDAR, using the output folder of pipeline/fetch/fetch_site.py.

    python analyze_site.py data/<slug>            -> writes data/<slug>/scene.json

Model frame: feet, +x east, +z south, origin at site.json mercator_center (the geocode),
y relative to PAD (terrain elevation at the origin, NAVD88 feet).
Needs: numpy, scipy, laspy[lazrs].
"""
import argparse, glob, json, math, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FT = 3.28084
NONBUILD = {2, 7, 9, 17, 18}
MAX_BLDG_FT = 1800.0                  # above every US building; noise classes 7/18 are dropped separately          # ground, noise, water, bridge deck, high noise
rng = np.random.default_rng(7)


# ---------------------------------------------------------------- inputs
def dem_bounds(dem):
    """[xmin, xmax, zmin, zmax] outer cell edges in the model frame (Codex's dem.json bounds_ft; may be
    asymmetric). Legacy files without it are centred on the origin."""
    c, R, C = dem['cell_size_ft'], dem['rows'], dem.get('cols', dem['rows'])
    b = dem.get('bounds_ft')
    if b:
        assert abs((b[1] - b[0]) - C * c) < 0.05 and abs((b[3] - b[2]) - R * c) < 0.05, 'dem.json bounds_ft disagree with rows/cols'
        return [float(v) for v in b]
    return [-C * c / 2, C * c / 2, -R * c / 2, R * c / 2]


def load_site(folder):
    site = json.load(open(os.path.join(folder, 'site.json')))
    if site.get('status') != 'complete':
        sys.exit(f"site.json status is {site.get('status')!r}, expected 'complete'")
    dem = json.load(open(os.path.join(folder, 'dem.json')))
    R, C, c = dem['rows'], dem.get('cols', dem['rows']), dem['cell_size_ft']
    E = np.array(dem['elevations_ft'], float).reshape(R, C)
    roads = None
    p = os.path.join(folder, 'roads.json')
    if os.path.exists(p):
        r = json.load(open(p))
        roads = r.get('roads', []) if isinstance(r, dict) else r
    return site, E, c, roads, dem_bounds(dem)


class Terrain:
    """Bilinear bare-earth height (relative to PAD) anywhere in the DEM; clamps at its edge."""
    def __init__(self, E, cell, bounds=None):
        self.E, self.c = E, cell
        self.R, self.C = E.shape
        self.n = self.R
        self.b = bounds or [-self.C * cell / 2, self.C * cell / 2, -self.R * cell / 2, self.R * cell / 2]
        self.pad = 0.0
        self.pad = float(self(0.0, 0.0))

    def __call__(self, x, z):
        x = np.asarray(x, float); z = np.asarray(z, float)
        fx = np.clip((x - self.b[0]) / self.c - .5, 0, self.C - 1.001)
        fz = np.clip((z - self.b[2]) / self.c - .5, 0, self.R - 1.001)
        j = np.floor(fx).astype(int); i = np.floor(fz).astype(int); u = fx - j; v = fz - i
        E = self.E
        h = E[i, j] * (1 - u) * (1 - v) + E[i, j + 1] * u * (1 - v) + E[i + 1, j] * (1 - u) * v + E[i + 1, j + 1] * u * v
        return h - self.pad


def dataset_year(name):
    m = re.search(r'(20\d\d|19\d\d)', name)
    if m: return int(m.group(1))
    m = re.search(r'_D(\d\d)\b', name)
    return 2000 + int(m.group(1)) if m else 0


def load_lidar(folder, site, terrain, bounds=None):
    import laspy
    cx, cy = site['mercator_center']; k = math.cos(math.radians(site['lat'])) * FT
    man = json.load(open(os.path.join(folder, 'lidar', 'manifest.json')))
    out = []
    for name, info in man['datasets'].items():
        if 'error' in info: continue
        tiles = glob.glob(os.path.join(folder, 'lidar', name, 'ept-data', '*.laz'))
        rows = []
        for f in tiles:
            l = laspy.read(f)
            x = (np.asarray(l.x) - cx) * k; z = -(np.asarray(l.y) - cy) * k
            b = bounds or terrain.b
            m = (x >= b[0]) & (x <= b[1]) & (z >= b[2]) & (z <= b[3])
            if m.any():
                rows.append(np.c_[x[m], np.asarray(l.z)[m], z[m], np.asarray(l.classification)[m],
                                  np.asarray(l.number_of_returns)[m], np.asarray(l.intensity)[m]])
        if not rows: continue
        P = np.vstack(rows)
        g = P[:, 3] == 2
        T = terrain(P[:, 0], P[:, 2])
        # vertical units: EPT for 3DEP is metres; check against the DEM and pick the better fit
        best = None
        for scale in (FT, 1.0):
            if g.sum() < 50: off = 0; spread = 0
            else:
                d = P[g, 1] * scale - terrain.pad - T[g]
                off = float(np.median(d)); spread = float(np.percentile(np.abs(d - off), 75))
            if best is None or spread < best[2]: best = (scale, off, spread)
        scale, off, spread = best
        y = P[:, 1] * scale - terrain.pad - off
        out.append(dict(name=name, year=dataset_year(name), x=P[:, 0], y=y, z=P[:, 2], cls=P[:, 3].astype(int),
                        nret=P[:, 4].astype(int), inten=P[:, 5], hag=y - T, offset_ft=off, spread_ft=spread, n=len(P)))
    if not out: sys.exit('no usable LiDAR datasets in lidar/manifest.json')
    out.sort(key=lambda d: -d['year'])
    return out


# ---------------------------------------------------------------- geometry helpers
def ransac_planes(P, thr=0.7, min_in=25, iters=1200, max_planes=10):
    planes = []; rem = np.arange(len(P))
    for _ in range(max_planes):
        if len(rem) < min_in: break
        Q = P[rem]
        idx = rng.integers(0, len(Q), size=(iters, 3))
        a, b, c = Q[idx[:, 0]], Q[idx[:, 1]], Q[idx[:, 2]]
        nrm = np.cross(b - a, c - a); nn = np.linalg.norm(nrm, axis=1); ok = nn > 1e-6
        nrm = nrm[ok] / nn[ok, None]; a = a[ok]
        ok2 = np.abs(nrm[:, 1]) > math.cos(math.radians(60))
        nrm, a = nrm[ok2], a[ok2]
        if len(nrm) == 0: break
        best_n, best_mask = 0, None
        for s in range(0, len(nrm), 200):
            D = np.abs(np.einsum('nkj,nj->nk', Q[None, :, :] - a[s:s + 200, None, :], nrm[s:s + 200]))
            cnt = (D < thr).sum(1); j = int(cnt.argmax())
            if cnt[j] > best_n: best_n, best_mask = int(cnt[j]), D[j] < thr
        if best_n < min_in: break
        I = rem[best_mask]
        for _ in range(2):
            X = np.c_[P[I, 0], P[I, 2], np.ones(len(I))]
            coef, *_ = np.linalg.lstsq(X, P[I, 1], rcond=None)
            r = np.abs(np.c_[P[rem, 0], P[rem, 2], np.ones(len(rem))] @ coef - P[rem, 1])
            I = rem[r < thr]
        if len(I) < min_in: break
        planes.append((coef, I)); rem = np.setdiff1d(rem, I)
    return planes


def plane_info(coef):
    a, b, _ = coef
    slope = math.hypot(a, b)
    return dict(pitch=math.degrees(math.atan(slope)), down=(-a / slope, -b / slope) if slope > 1e-6 else (0.0, 0.0))


def min_rect(x, z):
    best = None
    for ang in np.radians(np.arange(0, 90, 0.5)):
        u = x * math.cos(ang) + z * math.sin(ang); v = -x * math.sin(ang) + z * math.cos(ang)
        lu, hu = np.percentile(u, [1.5, 98.5]); lv, hv = np.percentile(v, [1.5, 98.5])
        A = (hu - lu) * (hv - lv)
        if best is None or A < best[0]: best = (A, ang, lu, hu, lv, hv)
    A, ang, lu, hu, lv, hv = best
    uc, vc = (lu + hu) / 2, (lv + hv) / 2
    cx = uc * math.cos(ang) - vc * math.sin(ang); cz = uc * math.sin(ang) + vc * math.cos(ang)
    return dict(area=A, ang=ang, cx=cx, cz=cz, lu=hu - lu, lv=hv - lv)


def to_local(x, z, cx, cz, r):
    x = np.asarray(x) - cx; z = np.asarray(z) - cz
    return x * math.cos(r) - z * math.sin(r), x * math.sin(r) + z * math.cos(r)


def to_world(lx, lz, cx, cz, r):
    return cx + lx * math.cos(r) + lz * math.sin(r), cz - lx * math.sin(r) + lz * math.cos(r)


# ---------------------------------------------------------------- roads
def road_polylines(roads):
    out = []
    for w in roads or []:
        pts = np.array(w.get('points') or w.get('pts') or [], float)
        if len(pts) >= 2: out.append((w, pts[:, :2]))
    return out


def nearest_road(pt, polys):
    best = None
    for w, P in polys:
        for a, b in zip(P[:-1], P[1:]):
            ab = b - a; L = float(ab @ ab)
            t = 0.0 if L == 0 else float(np.clip((pt - a) @ ab / L, 0, 1))
            q = a + t * ab; d = float(np.hypot(*(pt - q)))
            if best is None or d < best[0]:
                best = (d, q, ab / (math.sqrt(L) or 1), w)
    return best


# ---------------------------------------------------------------- clustering
def clusters(x, z, cell=2.0, dilate=1):
    from scipy import ndimage
    x0, z0 = x.min(), z.min()
    i = ((z - z0) / cell).astype(int); j = ((x - x0) / cell).astype(int)
    G = np.zeros((i.max() + 1, j.max() + 1), bool); G[i, j] = True
    if dilate: G = ndimage.binary_dilation(G, iterations=dilate)
    L, n = ndimage.label(G, structure=np.ones((3, 3)))
    return L[i, j], n


# ---------------------------------------------------------------- main analysis
def measure_building(P, planes, terrain, front_hint):
    """P: Nx3 (x,y,z) roof points of one building, planes from ransac. Returns dict in model terms."""
    roofish = [(c, I) for c, I in planes if plane_info(c)['pitch'] < 55]
    if not roofish: return None
    roofish.sort(key=lambda t: -len(t[1]))
    # main roof: the largest sloped plane, unless sloped planes are a minor part of the roof (then flat)
    sloped = [(c, I) for c, I in roofish if plane_info(c)['pitch'] >= 6 and len(I) >= 25]
    nsl = sum(len(I) for _, I in sloped); ntot = sum(len(I) for _, I in roofish)
    mc, mI = sloped[0] if sloped and nsl >= 0.3 * ntot else roofish[0]
    md = plane_info(mc)['down']
    if md == (0.0, 0.0):
        R0 = min_rect(P[mI, 0], P[mI, 2]); md = (math.cos(R0['ang']), math.sin(R0['ang']))
    axes = [md, (-md[1], md[0]), (-md[0], -md[1]), (md[1], -md[0])]
    f = max(axes, key=lambda a: a[0] * front_hint[0] + a[1] * front_hint[1])
    rot = math.atan2(f[0], f[1])          # three.js rotation.y so that local +z = f
    # footprint from the upper roof only (planes reaching above the main eave); porch and deck roofs excluded
    meave = float(np.percentile(P[mI, 1], 2))
    upper = [I for c, I in roofish if (np.dot(plane_info(c)['down'], md) < -0.8 and abs(plane_info(c)['pitch'] - plane_info(mc)['pitch']) < 12) or I is mI]
    allI = np.concatenate(upper)
    cx, cz = float(np.median(P[mI, 0])), float(np.median(P[mI, 2]))
    lx, lz = to_local(P[allI, 0], P[allI, 2], cx, cz, rot)
    x0, x1 = np.percentile(lx, [1.5, 98.5]); z0, z1 = np.percentile(lz, [1.5, 98.5])
    W = float(x1 - x0); D = float(z1 - z0)
    cx, cz = to_world((x0 + x1) / 2, (z0 + z1) / 2, cx, cz, rot)

    def loc_dir(d):
        return d[0] * math.cos(rot) - d[1] * math.sin(rot), d[0] * math.sin(rot) + d[1] * math.cos(rot)
    info = []
    for c, I in roofish:
        pi = plane_info(c); ld = loc_dir(pi['down'])
        px, pz = to_local(P[I, 0], P[I, 2], cx, cz, rot)
        info.append(dict(coef=c, I=I, n=len(I), pitch=pi['pitch'], ld=ld, px=px, pz=pz,
                         y2=float(np.percentile(P[I, 1], 2)), y98=float(np.percentile(P[I, 1], 98))))
    info.sort(key=lambda e: -e['n'])
    main_e = next(e for e in info if e['I'] is mI)
    info = [main_e] + [e for e in info if e is not main_e]
    flat = main_e['pitch'] < 6
    overhang = 1.0
    w = max(16.0, round(W - 2 * overhang, 1)); d = max(16.0, round(D - 2 * overhang, 1))
    grade = float(terrain(*to_world(0, d / 2 + 3, cx, cz, rot)))
    out = dict(x=round(cx, 2), z=round(cz, 2), rot=round(rot, 4), w=w, d=d, grade=round(grade, 2))
    if flat:
        top = info[0]['y98']
        out.update(roof=dict(type='flat', pitch_deg=0), wall_h=round(max(9, top - grade - 1), 1), ridge_h=round(top - grade, 1))
        out['stories'] = 1 if out['wall_h'] < 14 else 2
        return out, info
    main = info[0]
    # partner plane: opposite downslope
    partners = [e for e in info[1:] if e['pitch'] > 6 and (e['ld'][0] * main['ld'][0] + e['ld'][1] * main['ld'][1]) < -0.8]
    along_x = abs(main['ld'][1]) > abs(main['ld'][0])   # main slopes toward +-z => ridge along local x
    pitch = main['pitch'] if not partners else (main['pitch'] * main['n'] + partners[0]['pitch'] * partners[0]['n']) / (main['n'] + partners[0]['n'])
    pitch = float(np.clip(pitch, 8, 50))
    mains = [main] + partners[:1]
    eave = float(np.median([e['y2'] for e in mains]))
    span = d if along_x else w
    ridge_fit = max(e['y98'] for e in mains)
    ridge_geom = eave + (span / 2 + overhang) * math.tan(math.radians(pitch))
    ridge = ridge_fit if abs(ridge_fit - ridge_geom) < 1.5 else (ridge_fit + ridge_geom) / 2
    # ends: planes sloping along the ridge axis => hip ends or cross gables
    pid = {id(e) for e in partners}
    ends = [e for e in info if e is not main and id(e) not in pid and e['pitch'] > 12 and
            abs(e['ld'][0] * main['ld'][0] + e['ld'][1] * main['ld'][1]) < 0.35 and e['n'] >= 20]
    hip = False; cross = []
    if ends:
        axis_c = (lambda e: e['px']) if along_x else (lambda e: e['pz'])
        L = (w if along_x else d) / 2
        for e in ends:
            # a hip end hugs the end of the main roof and slopes outward; a cross gable sits inside the span
            c = float(np.median(axis_c(e))); outward = (e['ld'][0] if along_x else e['ld'][1]) * np.sign(c)
            if abs(c) > 0.6 * L and outward > 0.5 and abs(e['pitch'] - pitch) < 10: e['hip'] = True; hip = True
        pairs = []; used = set()
        rest = [e for e in ends if not e.get('hip')]
        for i, a in enumerate(rest):
            for j, b in enumerate(rest):
                if j <= i or i in used or j in used: continue
                if (a['ld'][0] * b['ld'][0] + a['ld'][1] * b['ld'][1]) < -0.8:
                    pairs.append((a, b)); used |= {i, j}
        for a, b in pairs:
            cpx = np.r_[a['px'], b['px']] if along_x else np.r_[a['pz'], b['pz']]
            cpz = np.r_[a['pz'], b['pz']] if along_x else np.r_[a['px'], b['px']]
            lo, hi = np.percentile(cpx, [3, 97]); zlo, zhi = np.percentile(cpz, [3, 97])
            top = max(a['y98'], b['y98']); cp = (a['pitch'] + b['pitch']) / 2
            cspan = float(min(hi - lo, (w if along_x else d) * 0.8)); cspan = max(8.0, cspan - 2 * overhang)
            ch = float(np.clip(top - eave, 3, 16))
            if ch < 5.5: continue
            c_new = dict(pos=round(float((lo + hi) / 2), 1), span=round(cspan, 1),
                         from_=round(float(zlo), 1), to=round(float(zhi), 1), h=round(ch, 1), pitch_deg=round(cp, 1))
            dup = next((c for c in cross if abs(c['pos'] - c_new['pos']) < max(c['span'], c_new['span']) / 2), None)
            if dup is None: cross.append(c_new)
            elif c_new['h'] > dup['h']: cross[cross.index(dup)] = c_new
    wall_h = round(max(8.0, eave - grade - 0.3), 1)
    out.update(wall_h=wall_h, ridge_h=round(ridge - grade, 1), eave_h=round(eave - grade, 1),
               roof=dict(type='hip' if hip else 'gable', ridge_axis='x' if along_x else 'z', pitch_deg=round(pitch, 1),
                         h=round(max(3.0, ridge - eave), 1), overhang=overhang,
                         cross=[{**{k: v for k, v in c.items() if k != 'from_'}, 'from': c['from_']} for c in cross]))
    out['stories'] = 1 if wall_h < 13 else (2 if wall_h < 23 else 3)
    return out, info


def _local_planes(hgt, mask, GX, GZ, win=5):
    """Least-squares plane over a win x win window at every cell: slope (a, b), offset, rms residual."""
    from scipy import ndimage
    w = mask.astype(float)
    f = lambda v: ndimage.uniform_filter(v * w, size=win, mode='constant')
    n = f(np.ones_like(hgt)); X = GX; Z = GZ; Hh = np.nan_to_num(hgt)
    Sx, Sz, Sh = f(X), f(Z), f(Hh); Sxx, Szz, Sxz = f(X * X), f(Z * Z), f(X * Z); Sxh, Szh, Shh = f(X * Hh), f(Z * Hh), f(Hh * Hh)
    M = np.stack([np.stack([Sxx, Sxz, Sx], -1), np.stack([Sxz, Szz, Sz], -1), np.stack([Sx, Sz, n], -1)], -2)
    r = np.stack([Sxh, Szh, Sh], -1)
    M = M + np.eye(3) * 1e-6
    ok = n * win * win > 6
    sol = np.zeros(hgt.shape + (3,))
    sol[ok] = np.linalg.solve(M[ok], r[ok][..., None])[..., 0]
    a, b, c = sol[..., 0], sol[..., 1], sol[..., 2]
    # residual: E[(h - aX - bZ - c)^2]
    res = (Shh - 2 * (a * Sxh + b * Szh + c * Sh) + a * a * Sxx + b * b * Szz + c * c * n + 2 * (a * b * Sxz + a * c * Sx + b * c * Sz)) / np.maximum(n, 1e-6)
    return a, b, c, np.sqrt(np.clip(res, 0, None)), ok


def segment_heights(hgt, mask, GX, GZ):
    """Split a roof height raster into planes by region growing (seeded at the flattest cells). Returns
    (label raster with 1-based plane ids, 0 outside mask; list of [a, b, c] with y = a*u + b*v + c)."""
    from scipy import ndimage
    from collections import deque
    shape = hgt.shape
    la, lb, lc, lres, lok = _local_planes(hgt, mask, GX, GZ)
    label = np.zeros(shape, int); planes = []
    order = np.argsort(np.where(mask & lok, lres, 1e9), axis=None)
    nbr = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for flat in order:
        si, sj = divmod(int(flat), shape[1])
        if not mask[si, sj] or label[si, sj] or lres[si, sj] > 0.6: continue
        k = len(planes) + 1; pa, pb = la[si, sj], lb[si, sj]; pc = hgt[si, sj] - pa * GX[si, sj] - pb * GZ[si, sj]
        reg = [(si, sj)]; label[si, sj] = k; q = deque([(si, sj)]); added = 0
        while q:
            ci, cj = q.popleft()
            for di, dj in nbr:
                ni, nj = ci + di, cj + dj
                if ni < 0 or nj < 0 or ni >= shape[0] or nj >= shape[1]: continue
                if not mask[ni, nj] or label[ni, nj]: continue
                if abs(hgt[ni, nj] - (pa * GX[ni, nj] + pb * GZ[ni, nj] + pc)) > 1.0: continue
                if lres[ni, nj] < 0.6 and abs(la[ni, nj] - pa) + abs(lb[ni, nj] - pb) > 0.16: continue
                label[ni, nj] = k; reg.append((ni, nj)); q.append((ni, nj)); added += 1
                if added % 25 == 0 and len(reg) >= 8:
                    R = np.array(reg); A = np.c_[GX[R[:, 0], R[:, 1]], GZ[R[:, 0], R[:, 1]], np.ones(len(R))]
                    pa, pb, pc = np.linalg.lstsq(A, hgt[R[:, 0], R[:, 1]], rcond=None)[0]
        if len(reg) < 10:
            for ri, rj in reg: label[ri, rj] = 0
            continue
        R = np.array(reg); A = np.c_[GX[R[:, 0], R[:, 1]], GZ[R[:, 0], R[:, 1]], np.ones(len(R))]
        pa, pb, pc = np.linalg.lstsq(A, hgt[R[:, 0], R[:, 1]], rcond=None)[0]
        planes.append([float(pa), float(pb), float(pc)])
    # leftovers (ridges, clutter): join the adjacent facet whose plane fits best, else the nearest facet
    for _ in range(6):
        left = mask & (label == 0)
        if not left.any() or not planes: break
        for li, lj in zip(*np.nonzero(left)):
            best = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = li + di, lj + dj
                    if 0 <= ni < shape[0] and 0 <= nj < shape[1] and label[ni, nj] > 0:
                        pa, pb, pc = planes[label[ni, nj] - 1]
                        e = abs(hgt[li, lj] - (pa * GX[li, lj] + pb * GZ[li, lj] + pc))
                        if best is None or e < best[0]: best = (e, label[ni, nj])
            if best: label[li, lj] = best[1]
    if (mask & (label == 0)).any() and planes:
        idx = ndimage.distance_transform_edt(label == 0, return_distances=False, return_indices=True)
        label = np.where(mask, label[tuple(idx)], 0)
    if not planes:
        planes = [[0.0, 0.0, float(np.median(hgt[mask]))]]; label = np.where(mask, 1, 0)
    for _ in range(3):
        changed = False
        for k in range(1, len(planes) + 1):
            R = label == k; n = int(R.sum())
            if n == 0 or n >= 20: continue
            ring = ndimage.binary_dilation(R) & ~R & mask
            cand = [l for l in np.unique(label[ring]) if l > 0 and l != k]
            if not cand: continue
            def fit(l):
                pa, pb, pc = planes[l - 1]
                return float(np.median(np.abs(hgt[R] - (pa * GX[R] + pb * GZ[R] + pc))))
            best = min(cand, key=fit)
            if fit(best) < 3.0 or n < 8: label[R] = best; changed = True
        if not changed: break
    return label, planes


def roofgrid(c, terr, cell=2.0, zone=None):
    """Any building that is not a simple house: segment the roof into planes by region growing on a
    height raster aligned with the building's main axis, and return the labelled grid. The page builds
    one continuous roof surface from it (each cell on its plane), with walls on the outline and at height
    steps, so ridges are sharp and there are no cracks. zone(x, z) -> bool clips very large buildings."""
    from scipy import ndimage
    from skimage import measure, morphology
    from collections import deque
    Pw = c['P']
    if zone is not None:
        keep = zone(Pw[:, 0], Pw[:, 2])
        if keep.sum() >= 30: Pw = Pw[keep]
    # raster aligned with the building's main axis, so walls come out straight and corners square
    ang = c['rect']['ang']; ca, sa = math.cos(ang), math.sin(ang)
    P = np.c_[Pw[:, 0] * ca + Pw[:, 2] * sa, Pw[:, 1], -Pw[:, 0] * sa + Pw[:, 2] * ca]
    x0, z0 = float(P[:, 0].min()) - 2 * cell, float(P[:, 2].min()) - 2 * cell
    j = ((P[:, 0] - x0) / cell).astype(int); i = ((P[:, 2] - z0) / cell).astype(int)
    shape = (i.max() + 3, j.max() + 3)
    occ = np.zeros(shape, bool); occ[i, j] = True
    mask = ndimage.binary_closing(occ, iterations=1) | occ
    mask = morphology.remove_small_holes(mask, max_size=60)
    mask = morphology.remove_small_objects(mask, max_size=24)
    sums = np.zeros(shape); cnt = np.zeros(shape)
    np.add.at(sums, (i, j), P[:, 1]); np.add.at(cnt, (i, j), 1)
    hgt = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    idx = ndimage.distance_transform_edt(np.isnan(hgt), return_distances=False, return_indices=True)
    hgt = hgt[tuple(idx)]
    hgt = np.where(mask, ndimage.median_filter(hgt, size=3), hgt)
    gx = x0 + (np.arange(shape[1]) + .5) * cell; gz = z0 + (np.arange(shape[0]) + .5) * cell
    GX, GZ = np.meshgrid(gx, gz)
    label, planes = segment_heights(hgt, mask, GX, GZ)
    # upsample x2 with a smoothed outline, so angled walls step in 1 ft, not 2-2.5 ft
    f_ = 2
    lab_up = np.kron(label, np.ones((f_, f_), int))
    m_up = ndimage.gaussian_filter(np.kron(mask.astype(float), np.ones((f_, f_))), sigma=f_ * 0.9) > 0.5
    if (m_up & (lab_up == 0)).any():
        idx_ = ndimage.distance_transform_edt(lab_up == 0, return_distances=False, return_indices=True)
        lab_up = np.where(m_up, lab_up[tuple(idx_)], 0)
    else:
        lab_up = np.where(m_up, lab_up, 0)
    fine = cell / f_
    # run-length rows of plane labels; planes stay in the building's (u, v) frame
    label_out, shape_out = lab_up, lab_up.shape
    rows = []
    for r in range(shape_out[0]):
        lab_r = label_out[r]; runs = []; k = 0
        while k < shape_out[1]:
            if lab_r[k] > 0:
                s0, L = k, lab_r[k]
                while k < shape_out[1] and lab_r[k] == L: k += 1
                runs += [int(s0), int(k - s0), int(L - 1)]
            else: k += 1
        rows.append(runs)
    us, vs = GX[mask], GZ[mask]
    xs, zs = us * ca - vs * sa, us * sa + vs * ca
    tops = np.array([planes[l - 1][0] * u + planes[l - 1][1] * v + planes[l - 1][2] for u, v, l in zip(us, vs, label[mask])])
    grade = float(np.median(terr(xs, zs)))
    slopes = [math.degrees(math.atan(math.hypot(a_, b_))) for a_, b_, _ in planes]
    counts = np.bincount(label[mask], minlength=len(planes) + 1)[1:]
    pitched = [(sl, n_) for sl, n_ in zip(slopes, counts) if sl > 8 and n_ > 0]   # planes can lose all cells to merging
    pitch = float(np.average([p_ for p_, _ in pitched], weights=[n_ for _, n_ in pitched])) if pitched else 0.0
    eave = float(np.percentile(tops, 20) - grade); top = float(np.percentile(tops, 99.5) - grade)
    return dict(type='grid', x=round(float(xs.mean()), 2), z=round(float(zs.mean()), 2), rot=round(-ang, 4),
                grade=round(grade, 2), w=round(float(np.ptp(us) + cell), 1), d=round(float(np.ptp(vs) + cell), 1),
                wall_h=round(eave, 1), eave_h=round(eave, 1), ridge_h=round(top, 1), area_ft2=round(float(mask.sum() * cell * cell)),
                stories=max(1, int(round(eave / 11))),
                roof=dict(type='grid', pitch_deg=round(pitch, 1), planes=len(planes)),
                grid=dict(ang=round(ang, 6), x0=round(x0, 3), z0=round(z0, 3), cell=fine, rows=rows,
                          planes=[[round(a_, 5), round(b_, 5), round(c_, 3)] for a_, b_, c_ in planes]))


def pavement(prim, G):
    """Last-resort paving estimate from LiDAR return intensity, used only when no mapped roads or surfaces
    exist. Raw intensity is uncalibrated (range, angle and sensor all change it), so the page labels this
    an estimate. Returns a run-length-encoded mask on the site grid G."""
    from scipy import ndimage
    from skimage import filters, morphology
    g = prim['cls'] == 2
    if g.sum() < 500: return None
    i, j, ok = G.ij(prim['x'][g], prim['z'][g]); it = prim['inten'][g].astype(float)
    s_ = np.zeros((G.R, G.C)); c_ = np.zeros((G.R, G.C))
    np.add.at(s_, (i[ok], j[ok]), np.log1p(it[ok])); np.add.at(c_, (i[ok], j[ok]), 1)
    has = c_ > 0
    if has.sum() < 200: return None
    v = np.where(has, s_ / np.maximum(c_, 1), np.nan)
    thr = filters.threshold_otsu(v[has])
    lo, hi = np.percentile(v[has], [10, 90])
    if hi - lo < 0.6: return None
    dark = has & (v < thr)
    veg = (~np.isin(prim['cls'], [2, 7, 9, 17, 18])) & (prim['nret'] > 1) & (prim['hag'] > 3)
    vi, vj, vk = G.ij(prim['x'][veg], prim['z'][veg])
    canopy = np.zeros((G.R, G.C), bool); canopy[vi[vk], vj[vk]] = True
    dark &= ~ndimage.binary_dilation(canopy, iterations=1)
    dark = ndimage.median_filter(dark.astype(np.uint8), size=3).astype(bool)
    dark = morphology.remove_small_objects(dark, max_size=40)
    dark = morphology.remove_small_holes(dark, max_size=30)
    pavement.mask = dark
    return dict(G.meta(), rle=rle(dark), share=round(float(dark.mean()), 3))


class Grid:
    """The DEM rectangle as a raster (rows north to south, columns west to east), for the site zone and
    pavement. bounds: [xmin, xmax, zmin, zmax] outer edges in the model frame."""
    def __init__(self, bounds, cell=2.65):
        self.b = list(bounds)
        self.C = max(1, int(round((bounds[1] - bounds[0]) / cell))); self.R = max(1, int(round((bounds[3] - bounds[2]) / cell)))
        self.c = (bounds[1] - bounds[0]) / self.C
        self.n = self.R

    def ij(self, x, z):
        j = np.floor((np.asarray(x) - self.b[0]) / self.c).astype(int); i = np.floor((np.asarray(z) - self.b[2]) / self.c).astype(int)
        return i, j, (i >= 0) & (j >= 0) & (i < self.R) & (j < self.C)

    def raster(self, x, z):
        i, j, ok = self.ij(x, z); M = np.zeros((self.R, self.C), bool); M[i[ok], j[ok]] = True; return M

    def lookup(self, M, x, z):
        i, j, ok = self.ij(x, z); out = np.zeros(len(np.atleast_1d(x)), bool)
        out[ok] = M[i[ok], j[ok]]; return out

    def xz(self, i, j):
        return self.b[0] + (np.asarray(j) + .5) * self.c, self.b[2] + (np.asarray(i) + .5) * self.c

    def centers(self):
        gx, gz = self.xz(np.arange(self.R)[:, None], np.arange(self.C)[None, :])
        return np.broadcast_to(gx, (self.R, self.C)), np.broadcast_to(gz, (self.R, self.C))

    def meta(self):
        return dict(rows=self.R, cols=self.C, cell=round(self.c, 5), bounds=[round(v, 3) for v in self.b])


def rle(M):
    rows = []
    for row in M:
        runs = []; k = 0; n = len(row)
        while k < n:
            if row[k]:
                s0 = k
                while k < n and row[k]: k += 1
                runs += [int(s0), int(k - s0)]
            else: k += 1
        rows.append(runs)
    return rows


def house_like(c, hh):
    """A simple detached house: small footprint, a dominant main roof pair, house-range heights."""
    if hh is None or c['rect']['area'] > 3500: return False
    if hh['roof']['type'] == 'flat': return False
    roof = [J for cf, J in c['planes'] if plane_info(cf)['pitch'] < 55]
    tot = sum(len(J) for J in roof); top2 = sum(sorted((len(J) for J in roof), reverse=True)[:2])
    return (top2 >= 0.5 * tot and 8 <= hh['wall_h'] <= 26 and hh['ridge_h'] <= 42 and max(hh['w'], hh['d']) <= 80)


SUFFIX = {'avenue': 'ave', 'av': 'ave', 'street': 'st', 'road': 'rd', 'drive': 'dr', 'boulevard': 'blvd', 'lane': 'ln',
          'court': 'ct', 'circle': 'cir', 'place': 'pl', 'parkway': 'pkwy', 'terrace': 'ter', 'highway': 'hwy', 'square': 'sq',
          'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw'}
ORD = {'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th', 'sixth': '6th', 'seventh': '7th',
       'eighth': '8th', 'ninth': '9th', 'tenth': '10th', 'eleventh': '11th', 'twelfth': '12th'}


def _street_tokens(st):
    toks = re.findall(r'[a-z0-9]+', (st or '').lower())
    return frozenset(SUFFIX.get(ORD.get(t, t), ORD.get(t, t)) for t in toks)


def parse_address(addr):
    """'175 5th Ave, New York, NY 10010' -> ('175', {'5th', 'ave'})"""
    m = re.match(r'\s*(\d+[A-Za-z]?(?:-\d+)?)\s+([^,]+)', addr or '')
    return (m.group(1).lower(), _street_tokens(m.group(2))) if m else None


def address_match(want, tags):
    """House number equal (or inside a tagged range like '171-177') and the street names agree once suffixes,
    directions and ordinals are normalised (one side may omit the suffix or direction)."""
    if not want or not tags.get('addr:housenumber'): return False
    num, st = want
    hn = str(tags['addr:housenumber']).lower().replace(' ', '')
    ok_num = False
    for part in re.split(r'[;,]', hn):
        r = re.match(r'(\d+)-(\d+)$', part)
        if part == num or (r and num.isdigit() and int(r.group(1)) <= int(num) <= int(r.group(2))): ok_num = True
    if not ok_num: return False
    ts = _street_tokens(tags.get('addr:street'))
    if not ts or not st: return False
    dirs = {'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'}
    sufx = set(SUFFIX.values()) - dirs
    name = lambda t: t - sufx - dirs
    if name(ts) != name(st) or not name(ts): return False
    a, b = ts & sufx, st & sufx
    return not (a and b) or bool(a & b)          # a suffix may be omitted on one side, but must not conflict


def load_features(folder):
    """Mapped evidence in the model frame: features.json (schema 1, from pipeline/fetch), else the legacy
    osm_buildings.json + roads.json. Returns dict(buildings, surfaces, ways, entrances, coverage)."""
    p = os.path.join(folder, 'features.json')
    if os.path.exists(p):
        F = json.load(open(p))
        if F.get('schema_version') != 1: sys.exit(f"features.json schema {F.get('schema_version')} not supported")
        return F
    F = dict(buildings=[], surfaces=[], ways=[], entrances=[], coverage=None)
    p = os.path.join(folder, 'osm_buildings.json')
    if os.path.exists(p):
        d = json.load(open(p)); d = d.get('buildings', []) if isinstance(d, dict) else d
        for b in d:
            pts = b.get('points') or b.get('pts') or []
            if len(pts) >= 4:
                F['buildings'].append(dict(id=f"osm-w{b.get('id')}", polygon=pts[:-1] if pts[0] == pts[-1] else pts, holes=[],
                                           tags=b.get('tags') or {}, source='osm', source_id=f"way/{b.get('id')}", source_date=None, confidence=None))
    p = os.path.join(folder, 'roads.json')
    if os.path.exists(p):
        d = json.load(open(p)); d = d.get('roads', []) if isinstance(d, dict) else d
        for w in d:
            pts = w.get('points') or w.get('pts') or []
            if len(pts) >= 2:
                F['ways'].append(dict(kind=w.get('highway') or 'residential', centerline=pts, width_ft=None, width_source=None,
                                      name=w.get('name'), source='osm', source_id=f"way/{w.get('id')}"))
    return F


def lidar_outlines(G, x, z, H, NR, C, min_ft2=350):
    """Footprints where no map has them: connected single-return roof cells, contoured and simplified, so
    walls run straight instead of stepping cell by cell."""
    from scipy import ndimage
    from skimage import measure, morphology
    from shapely.geometry import Polygon
    m = (~np.isin(C, list(NONBUILD))) & (H > 6) & (H < MAX_BLDG_FT) & (NR <= 1)
    # roofs are locally planar, tree crowns are not (some surveys leave both unclassified). Per-cell height is the
    # LOWEST single return, so antennas, masts and birds above a roof do not break its planarity.
    i, j, ok = G.ij(x[m], z[m]); y = H[m]
    hg = np.full((G.R, G.C), np.inf); np.minimum.at(hg, (i[ok], j[ok]), y[ok])
    occ = np.isfinite(hg); hg = np.where(occ, hg, 0.0)
    GX, GZ = G.centers()
    _a, _b, _c, res, lok = _local_planes(hg, occ, np.asarray(GX), np.asarray(GZ))
    seeds = occ & lok & (res < 0.7)
    # tree cells: mostly multi-echo returns above 6 ft
    mt = (~np.isin(C, list(NONBUILD))) & (H > 6) & (H < MAX_BLDG_FT) & (NR > 1)
    ti, tj, tok = G.ij(x[mt], z[mt])
    tc = np.zeros((G.R, G.C)); np.add.at(tc, (ti[tok], tj[tok]), 1)
    sc = np.zeros((G.R, G.C)); np.add.at(sc, (i[ok], j[ok]), 1)
    tree = tc > sc
    # grow the planar seeds through occupied, non-tree cells: step walls between roof levels (a tower's tubes,
    # a wing against a taller block) fail the planarity test but belong to the same building
    M = ndimage.binary_dilation(seeds, iterations=3, mask=occ & ~tree) | seeds   # about 8 ft: bridges steps, not yards
    M = ndimage.binary_dilation(M, iterations=1) & occ
    # close only 1-cell gaps (about 2.6 ft): houses 8-15 ft apart must stay separate buildings
    M = ndimage.binary_closing(M, iterations=1) | M
    M = ndimage.binary_opening(ndimage.binary_fill_holes(M), iterations=1)
    M = morphology.remove_small_objects(M, max_size=int(min_ft2 / G.c ** 2))
    L, n = ndimage.label(M)
    out = []
    for k in range(1, n + 1):
        cs = measure.find_contours(np.pad(L == k, 1).astype(float), 0.5)
        if not cs: continue
        cnt = measure.approximate_polygon(max(cs, key=len), tolerance=1.0 / G.c)
        px, pz = G.xz(cnt[:, 0] - 1, cnt[:, 1] - 1)
        pg = Polygon(np.c_[px, pz]).buffer(0)
        if pg.area >= min_ft2: out.append(pg.simplify(1.2))
    return out


def analyze(folder, verbose=True, buffer_ft=100.0, whole_max_ft2=40000.0):
    """Site rule: the building under the address point, plus every building and ground feature that
    touches the zone within buffer_ft of its footprint. Touching buildings are kept whole up to
    whole_max_ft2, larger ones are cut at the zone with a section face. Outlines come from mapped
    footprints (features.json) where they exist and LiDAR supports them, else from LiDAR; heights and
    roof planes are always measured from LiDAR."""
    from scipy import ndimage
    import shapely
    from shapely.geometry import Polygon, Point, LineString, box as sbox
    from shapely.ops import unary_union
    import geometry as GEO
    site, E, cell, _roads, bounds = load_site(folder)
    terr = Terrain(E, cell, bounds)
    D = load_lidar(folder, site, terr, bounds)
    prim = D[0]
    G = Grid(bounds)
    F = load_features(folder)
    X = np.concatenate([d['x'] for d in D]); Y = np.concatenate([d['y'] for d in D]); Z = np.concatenate([d['z'] for d in D])
    C = np.concatenate([d['cls'] for d in D]); H = np.concatenate([d['hag'] for d in D])
    NR = np.concatenate([d['nret'] for d in D])
    S = np.concatenate([np.full(d['n'], i) for i, d in enumerate(D)])
    # where LiDAR and DEM both exist (features.json may report narrower LiDAR bounds)
    cov = F.get('coverage') or {}
    lb = cov.get('lidar_bounds_ft') or bounds
    dbox = sbox(max(bounds[0], lb[0]) + 2, max(bounds[2], lb[2]) + 2, min(bounds[1], lb[1]) - 2, min(bounds[3], lb[3]) - 2)
    nonb = (~np.isin(C, list(NONBUILD))) & (H > 6) & (H < MAX_BLDG_FT)
    # ---- footprints: mapped polygons with LiDAR support, else LiDAR outlines
    def mkpoly(b):
        try:
            pg = Polygon(b['polygon'], [h for h in (b.get('holes') or []) if len(h) >= 3]).buffer(0)
        except Exception: return None
        return pg if pg.area > 60 else None
    mapped = []
    for b in F.get('buildings', []):
        pg = mkpoly(b)
        if pg is not None and pg.intersects(dbox): mapped.append(dict(poly=pg, rec=b))
    single = nonb & (NR <= 1)
    roofpts = single
    def support(pg):
        """Share of the footprint whose cells have roof-like returns (single echo, > 6 ft above ground)."""
        Mf = G.raster(*_poly_cells(pg, G))
        Mr = G.raster(X[roofpts], Z[roofpts])
        n = Mf.sum()
        return float((Mf & ndimage.binary_dilation(Mr)).sum() / n) if n else 0.0
    for m in mapped: m['support'] = support(m['poly'])
    blds_src = [dict(poly=m['poly'], rec=m['rec'], support=m['support'], outline='mapped') for m in mapped]
    # LiDAR buildings no map covers (new construction, unmapped areas)
    mapped_union = unary_union([m['poly'].buffer(3) for m in mapped]) if mapped else None
    for pg in lidar_outlines(G, X, Z, H, NR, C):
        if mapped_union is not None and pg.intersection(mapped_union).area > 0.4 * pg.area: continue
        blds_src.append(dict(poly=pg, rec=None, support=1.0, outline='measured'))
    if not blds_src: sys.exit('no buildings found near this address (map or LiDAR)')
    # ---- the featured building. Evidence, strongest first: a footprint whose address tags match the requested
    # address (or fetch's target_building_id, which applies the same rule); the footprint under the address point
    # when the locator placed it on the structure; else the footprint under / nearest to the point.
    o = Point(0, 0)
    loc = json.load(open(os.path.join(folder, 'locate.json'))) if os.path.exists(os.path.join(folder, 'locate.json')) else {}
    precise = loc.get('precision', 'address-point') in ('address-point', 'given')
    want = parse_address(site.get('address', ''))
    tagged = [b for b in blds_src if b['rec'] and address_match(want, (b['rec'].get('tags') or {}))
              and b['poly'].distance(o) < (150 if precise else 600)]
    tid = cov.get('target_building_id')
    by_id = [b for b in blds_src if b['rec'] and tid and b['rec'].get('source_id') == tid and b['poly'].distance(o) < 600]
    inside = [b for b in blds_src if b['poly'].buffer(1.5).contains(o)]
    feat_basis = None
    if tagged: feat = min(tagged, key=lambda b: b['poly'].distance(o)); feat_basis = 'address tags'
    elif by_id: feat = by_id[0]; feat_basis = 'fetch target'
    elif inside: feat = min(inside, key=lambda b: b['poly'].area); feat_basis = 'address point'
    else:
        cand = [b for b in blds_src if b['support'] >= 0.25]
        feat = min(cand or blds_src, key=lambda b: b['poly'].distance(o))
        feat_basis = 'nearest'
        if feat['poly'].distance(o) > 120:
            sys.exit(f"no building within 120 ft of the address point (nearest is {feat['poly'].distance(o):.0f} ft away)")
    zone = feat['poly'].buffer(buffer_ft, quad_segs=16)
    inc = [b for b in blds_src if b['poly'].intersects(zone) and (b is feat or b['support'] >= 0.25 or b['outline'] == 'mapped')]
    # ---- roads for street direction and street-facing doors
    VEH = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street', 'service',
           'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
    rx = []
    for w in F.get('ways', []):
        if w['kind'] in VEH and w.get('service') not in ('driveway', 'parking_aisle'):
            ls = LineString(w['centerline'])
            rx += [ls.interpolate(t).coords[0] for t in np.arange(0, ls.length, 6.0)]
    roads_xy = np.array(rx) if rx else None
    def front_of(pg):
        c_ = pg.centroid
        if roads_xy is not None:
            d = np.hypot(roads_xy[:, 0] - c_.x, roads_xy[:, 1] - c_.y); k = int(np.argmin(d))
            v = roads_xy[k] - [c_.x, c_.y]; n = np.hypot(*v)
            if n > 1: return v / n
        return np.array([0.0, 1.0])
    street_name = None
    if roads_xy is not None:
        best = None
        for w in F.get('ways', []):
            if w['kind'] in VEH and w.get('name'):
                d = LineString(w['centerline']).distance(feat['poly'])
                if best is None or d < best[0]: best = (d, w['name'])
        street_name = best[1] if best else None
    nodes = F.get('entrances', [])
    # ---- paving: mapped ground wins; otherwise a LiDAR-intensity estimate, which also stands in for the street
    # direction (its largest component) and gives driveway evidence for garage doors (other components)
    has_map_ground = bool(F.get('ways') or F.get('surfaces'))
    pave = None if has_map_ground else pavement(prim, G)
    drive_xy = None; pave_roads_xy = None
    if pave is not None:
        from scipy import ndimage as _nd
        M_ = pavement.mask.copy()
        lab_, n_ = _nd.label(M_)
        if n_:
            sizes_ = _nd.sum(M_, lab_, range(1, n_ + 1)); big_ = int(np.argmax(sizes_)) + 1
            ii, jj = np.nonzero(M_); px_, pz_ = G.xz(ii, jj)
            drive_xy = np.c_[px_, pz_]
            if roads_xy is None:
                bi, bj = np.nonzero(lab_ == big_); bx_, bz_ = G.xz(bi, bj)
                pave_roads_xy = np.c_[bx_, bz_][::4]
    # ---- build every included building
    blds, edge_touch, cut_ids = [], [], []
    xs_all, zs_all = X[nonb], Z[nonb]
    # neighbourhood context for the type prior: the typical height and footprint of the OTHER buildings on the site
    ctx_h, ctx_a = [], []
    ys_all = Y[nonb]
    for b_ in inc:
        bx0, bz0, bx1, bz1 = b_['poly'].bounds
        sel_ = (xs_all > bx0) & (xs_all < bx1) & (zs_all > bz0) & (zs_all < bz1)
        ins_ = np.where(sel_)[0][shapely.contains_xy(b_['poly'], xs_all[sel_], zs_all[sel_])]
        if len(ins_) >= 20:
            c_ = b_['poly'].centroid
            ctx_h.append((float(np.percentile(ys_all[ins_], 98)) - float(terr(c_.x, c_.y)), b_['poly'].area))
        ctx_a.append(b_['poly'].area)
    def _wmed(pairs):                                      # area-weighted median: large buildings define the fabric
        pairs = sorted(pairs); tot = sum(a for _, a in pairs); acc = 0.0
        for v, a in pairs:
            acc += a
            if acc >= tot / 2: return v
    context = dict(buildings=len(inc), median_height_ft=round(float(_wmed(ctx_h)), 1) if ctx_h else None,
                   median_footprint_ft2=round(float(np.median(ctx_a))) if ctx_a else None,
                   note='height median is footprint-area weighted')
    for k, b in enumerate(sorted(inc, key=lambda b: b is not feat)):
        is_feat = b is feat
        bid = 'featured' if is_feat else f'b{len(blds) - 1 if blds else 0}'
        pg0 = b['poly']; pg = pg0; cut_lines = []
        if not is_feat and pg0.area > whole_max_ft2:
            pg = pg0.intersection(zone); cut_ids.append(bid)
        if not pg.within(dbox):
            pg = pg.intersection(dbox); edge_touch.append(bid)
        pg = max(GEO.as_polys(pg), key=lambda p: p.area, default=None)
        if pg is None or pg.area < 60: continue
        cut = pg.boundary.difference(pg0.boundary.buffer(0.3))
        if not cut.is_empty: cut_lines.append(cut)
        tags = (b['rec'] or {}).get('tags') or {}
        # points over this footprint (a little wider for eaves)
        bx0, bz0, bx1, bz1 = pg.buffer(3.5).bounds
        sel = (xs_all > bx0) & (xs_all < bx1) & (zs_all > bz0) & (zs_all < bz1)
        Ii = np.where(nonb)[0][sel]
        inn = shapely.contains_xy(pg.buffer(1.5 if b['outline'] == 'mapped' else 3.0), X[Ii], Z[Ii]); Ii = Ii[inn]
        Pp = np.c_[X[Ii], Y[Ii], Z[Ii], NR[Ii]]
        # Type prior (user, 2026-11-23): a building typed as a detached house uses the house model - per-house measurements
        # (roof type, pitch, cross gables, hip ends, block size) rendered by the detailed house builder. If the measurement
        # does not describe a simple house, or for any other type, the general solids method is used.
        hh = None
        btype = classify_type(tags, pg, Pp, terr, context)
        if btype['type'] == 'house' and pg.area <= 3500 and b['support'] >= 0.5 and not cut_lines:
            sP = Pp[Pp[:, 3] <= 1][:, :3]
            if len(sP) >= 60:
                planes = ransac_planes(sP)
                c_ = dict(P=sP, planes=planes, rect=min_rect(sP[:, 0], sP[:, 2]))
                res = measure_building(sP, planes, terr, front_of(pg))
                hh = res[0] if res else None
                if not house_like(c_, hh): hh = None
                if hh:
                    hh['outline_poly'] = [[round(x_, 2), round(z_, 2)] for x_, z_ in pg.exterior.coords[:-1]]
                    hh['building_type'] = btype
                    hh['model'] = dict(kind='house model', status='measured parameters, typical details',
                                       note='roof type, pitch, cross gables, hip ends and block size are measured from LiDAR; '
                                            'porch, garage, door, trim and siding are typical details for a house (defaults, not observed)')
        if hh is None:
            hh = mesh_building(pg, Pp, terr, b, tags, cut_lines, F.get('ways', []), nodes, roads_xy if roads_xy is not None else pave_roads_xy, G, GEO, drive_xy, 'mapped' if rx else 'paving-estimate', front_of(pg), context)
        hh.update(id=bid, role='featured' if is_feat else 'context', clipped=bid in cut_ids or bid in edge_touch,
                  provenance=dict(hh.get('provenance') or {}, outline=b['outline'] if b['rec'] or b['outline'] == 'measured' else 'measured',
                                  outline_source=(b['rec'] or {}).get('source_id'), lidar_support=round(b['support'], 2)))
        if tags.get('name'): hh['name'] = tags['name']
        if tags.get('addr:housenumber'): hh['number'] = tags['addr:housenumber']; hh['street'] = tags.get('addr:street')
        hh['use'] = tags.get('building')
        blds.append(hh)
    # ---- site = zone + whole footprints of the included buildings (with a little yard)
    gx, gz = G.centers()
    site_poly = unary_union([zone] + [b['poly'] for b in inc if b['poly'].area <= whole_max_ft2 or b is feat]).intersection(dbox.buffer(2))
    site_m = shapely.contains_xy(site_poly.buffer(4), gx, gz)
    # ---- trees: canopy peaks inside the site, not over any footprint
    allfp = unary_union([b['poly'].buffer(2.5) for b in blds_src])
    tm = (S == 0) & (~np.isin(C, list(NONBUILD))) & (H > 12) & (H < 140) & (NR > 1)
    tm &= G.lookup(site_m, X, Z)
    if tm.any(): tm[tm] &= ~shapely.contains_xy(allfp, X[tm], Z[tm])
    trees = []
    if tm.any():
        tx, tz, th = X[tm], Z[tm], H[tm]
        gi = np.floor(tx / 10).astype(int); gj = np.floor(tz / 10).astype(int)
        best = {}
        for a_, b_, hh_, xx, zz in zip(gi, gj, th, tx, tz):
            k_ = (a_, b_); cur = best.get(k_)
            if cur is None or hh_ > cur[0]: best[k_] = (hh_, xx, zz, (cur[3] if cur else 0) + 1)
            else: best[k_] = (cur[0], cur[1], cur[2], cur[3] + 1)
        for hh_, xx, zz, cnt in sorted(best.values(), key=lambda v: -v[0]):
            if cnt < 3: continue
            if any((xx - t['x']) ** 2 + (zz - t['z']) ** 2 < 14 ** 2 for t in trees): continue
            trees.append(dict(x=round(float(xx), 1), z=round(float(zz), 1), h=round(float(min(hh_, 120)), 1)))
            if len(trees) >= 600: break
    fp = feat['poly']
    outline = [[round(x_, 1), round(z_, 1)] for x_, z_ in fp.exterior.coords[:-1]]
    fh_ = next((h for h in blds if h['id'] == 'featured'), None)
    if fh_ and fh_.get('footprint'):
        outline = fh_['footprint']                         # the built footprint (template-regularised when chosen)
    elif fh_ and fh_['roof']['type'] in ('gable', 'hip', 'flat') and b_outline_is_measured(feat):
        # a measured (LiDAR) outline also takes in decks and porches; for a house, outline the measured body
        c_, s_ = math.cos(fh_['rot']), math.sin(fh_['rot']); hw, hd = fh_['w'] / 2 + 1, fh_['d'] / 2 + 1
        outline = [[round(fh_['x'] + lx * c_ + lz * s_, 1), round(fh_['z'] - lx * s_ + lz * c_, 1)]
                   for lx, lz in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))]
    # This analysis is the point-cloud verification that fetch's pre-selection defers to: geometric coverage is
    # decided here (fetch's pre-analysis status is kept as source_status). Map availability is reported apart.
    status = 'incomplete' if edge_touch else ('expanded' if (cov.get('expansions') or 0) > 0 else 'complete')
    xb0, zb0, xb1, zb1 = dbox.bounds
    coverage = dict(status=status, edge_touching=edge_touch, section_cuts=cut_ids,
                    data_bounds_ft=[round(xb0, 1), round(xb1, 1), round(zb0, 1), round(zb1, 1)],
                    source_status=cov.get('status'), expansions=cov.get('expansions') or 0)
    fs = F.get('status')
    maps = dict(status=fs or ('ok' if (F.get('buildings') or F.get('ways')) else 'none'), error=F.get('error'))
    if edge_touch:
        want = unary_union([zone] + [b['poly'] for b in inc if b['poly'].area <= whole_max_ft2 or b is feat]).buffer(30).bounds
        coverage['wanted_bounds_ft'] = [round(want[0], 1), round(want[2], 1), round(want[1], 1), round(want[3], 1)]
    front = front_of(fp)
    # ---- observations from imagery (optional; produced by pipeline/derive/*): facades per building, ground cover
    obs_dir = os.path.join(folder, 'observations')
    ground_obs = None
    if os.path.exists(os.path.join(obs_dir, 'looks.json')):
        LK = json.load(open(os.path.join(obs_dir, 'looks.json')))
        for h_ in blds:
            if h_['id'] in LK.get('buildings', {}):
                h_['look'] = dict(LK['buildings'][h_['id']], status=LK.get('status', 'observed'), source=LK.get('source'), method=LK.get('method'))
    site_shapes = json.load(open(os.path.join(obs_dir, 'site_shapes.json'))) if os.path.exists(os.path.join(obs_dir, 'site_shapes.json')) else None
    if os.path.exists(os.path.join(obs_dir, 'ground.json')):
        ground_obs = json.load(open(os.path.join(obs_dir, 'ground.json')))
        ground_obs['note'] = 'classes from the aerial; buildings masked by their footprints'
    scene = dict(schema_version=3, address=site['address'], matched_address=site.get('matched_address'),
                 lat=site['lat'], lon=site['lon'], pad_ft=round(terr.pad, 2), buffer_ft=buffer_ft,
                 dem_bounds_ft=bounds, coverage=coverage, maps=maps,
                 datasets=[dict(name=d['name'], year=d['year'], points=int(d['n']), z_offset_ft=round(d['offset_ft'], 2),
                                fit_spread_ft=round(d['spread_ft'], 2)) for d in D],
                 street=dict(front=[round(float(front[0]), 4), round(float(front[1]), 4)], name=street_name,
                             source='mapped' if rx else ('lidar-intensity' if pave else 'none')),
                 site=dict(G.meta(), rle=rle(site_m)), outline=outline, houses=blds, trees=trees, pavement=None if (ground_obs or site_shapes) else pave, ground_cover=ground_obs, site_shapes=site_shapes,
                 featured_basis=feat_basis, candidates=len(blds_src), footprints=dict(mapped=len(mapped), measured=sum(1 for b in blds_src if b['outline'] == 'measured')))
    # withheld roof points go to a sidecar for gates.py, never into the page
    ho = {hh['id']: hh.pop('_heldout') for hh in blds if '_heldout' in hh}
    json.dump(ho, open(os.path.join(folder, 'heldout.json'), 'w'), separators=(',', ':'))
    json.dump(scene, open(os.path.join(folder, 'scene.json'), 'w'), separators=(',', ':'))
    if verbose:
        for hh in blds:
            r = hh['roof']; pv = hh.get('provenance', {})
            print(f"{hh['id']:9s} {(hh.get('name') or hh.get('use') or '')[:28]:28s} ({hh['x']:7.1f},{hh['z']:7.1f}) eave {hh['wall_h']:5.1f} top {hh['ridge_h']:5.1f} "
                  f"{r['type']:5s} outline {pv.get('outline')} support {pv.get('lidar_support')} doors {len(hh.get('doors', []))} windows {len(hh.get('windows', []))}"
                  f"{' CLIPPED' if hh.get('clipped') else ''}")
        print(f"buildings {len(blds)}, trees {len(trees)}, street {scene['street']['source']} {street_name or ''}, "
              f"footprints {scene['footprints']}, coverage {status} {edge_touch or ''}, maps {maps['status']}, featured by {feat_basis}")
    return scene


def b_outline_is_measured(b):
    return b.get('outline') == 'measured'


def _poly_cells(pg, G):
    """Centres of site-grid cells inside a polygon (for rasterizing footprints)."""
    import shapely
    x0, z0, x1, z1 = pg.bounds
    i0, j0, _ = G.ij([x0], [z0]); i1, j1, _ = G.ij([x1], [z1])
    ii = np.arange(max(0, i0[0]), min(G.R, i1[0] + 1)); jj = np.arange(max(0, j0[0]), min(G.C, j1[0] + 1))
    if not len(ii) or not len(jj): return np.array([]), np.array([])
    gx, gz = G.xz(ii[:, None], jj[None, :]); gx, gz = np.broadcast_arrays(gx, gz)
    m = shapely.contains_xy(pg, gx, gz)
    return gx[m], gz[m]


def merge_coplanar(label, planes, hgt, mask, GU, GV, max_deg=8.0, max_med=0.6):
    """Region growing over-segments roofs (noise splits one plane into several). Merge planes whose normals
    are within max_deg of each other when one plane fits both regions (median residual under max_med ft).
    label: 0-based, all cells assigned. Returns relabelled (compact) label and planes."""
    P = [list(p) for p in planes]
    alive = list(range(len(P)))
    def normal(p): n = np.array([-p[0], -p[1], 1.0]); return n / np.linalg.norm(n)
    changed = True
    while changed:
        changed = False
        sizes = {k: int(((label == k) & mask).sum()) for k in alive}
        for a in sorted(alive, key=lambda k: -sizes[k]):
            if a not in alive: continue
            for b in sorted(alive, key=lambda k: -sizes[k]):
                if b == a or b not in alive: continue
                if math.degrees(math.acos(min(1.0, float(normal(P[a]) @ normal(P[b]))))) > max_deg: continue
                R = ((label == a) | (label == b)) & mask
                if R.sum() < 3: continue
                A = np.c_[GU[R], GV[R], np.ones(R.sum())]
                fit = np.linalg.lstsq(A, hgt[R], rcond=None)[0]
                if np.median(np.abs(A @ fit - hgt[R])) > max_med: continue
                P[a] = [float(v) for v in fit]; label[label == b] = a; alive.remove(b); changed = True
    remap = {k: i for i, k in enumerate(alive)}
    return np.vectorize(remap.get)(label), [P[k] for k in alive]


def absorb_fragments(L, min_cells):
    """Pieces of a plane smaller than min_cells join the neighbouring plane they share the longest edge with."""
    from scipy import ndimage
    L = L.copy()
    for _ in range(3):
        changed = False
        for k in np.unique(L):
            comp, n = ndimage.label(L == k)
            if n == 0: continue
            sizes = ndimage.sum(np.ones_like(L), comp, range(1, n + 1))
            for ci, sz in enumerate(sizes, 1):
                if sz >= min_cells: continue
                R = comp == ci
                ring = ndimage.binary_dilation(R) & ~R
                nb = L[ring]; nb = nb[nb != k]
                if len(nb): L[R] = np.bincount(nb).argmax(); changed = True
        if not changed: break
    return L


HOUSE_TAGS = {'house', 'detached', 'semidetached_house', 'bungalow', 'cabin', 'farm', 'residential_house'}
BLOCK_TAGS = {'apartments', 'commercial', 'retail', 'office', 'industrial', 'warehouse', 'school', 'university', 'college',
              'church', 'cathedral', 'hospital', 'hotel', 'civic', 'public', 'government', 'train_station', 'stadium', 'dormitory',
              'kindergarten', 'supermarket', 'parking', 'garages', 'terrace', 'townhouse'}


def classify_type(tags, pg, Pp, terr, context=None):
    """Building type prior from computable signals (user decision 2026-11-23). Map tags decide when specific; otherwise
    size and measured height. Returns an evidence record (C1). Neighbourhood context is not used yet."""
    t = (tags or {}).get('building')
    sig = dict(building_tag=t, area_ft2=round(pg.area))
    if t in HOUSE_TAGS: return dict(type='house', status='mapped', method='OSM building tag', signals=sig)
    if t in BLOCK_TAGS: return dict(type='block', status='mapped', method='OSM building tag', signals=sig)
    if len(Pp):
        g = float(np.median(terr(*np.array(pg.exterior.coords).T)))
        top = float(np.percentile(Pp[:, 1], 98)) - g; sig['height_ft'] = round(top, 1)
    else: top = None
    rect = pg.minimum_rotated_rectangle; sig['rectangularity'] = round(pg.area / max(rect.area, 1), 2)
    ctx = context or {}; sig['context'] = ctx
    residential_fabric = (ctx.get('median_height_ft') is None or ctx['median_height_ft'] <= 45) and \
                         (ctx.get('median_footprint_ft2') is None or ctx['median_footprint_ft2'] <= 5000)
    house = 500 <= pg.area <= 3500 and top is not None and 10 <= top <= 42 and residential_fabric
    return dict(type='house' if house else 'block', status='inferred',
                method='footprint 500-3500 ft2, measured height 10-42 ft, and neighbours of house scale (median height <= 45 ft, '
                       'median footprint <= 5000 ft2) => detached house; otherwise block (provisional rule)',
                signals=sig)


def typed_house_defaults(R, doors, front, terr):
    """Typical features of a detached house, drawn as LABELLED DEFAULTS (status 'default': typical for the type, not
    observed). Evidence replaces them: a garage door inferred from a driveway suppresses the default garage.
    - front door: a stoop (concrete pad and step);
    - garage: one door (two if the street wall is >= 44 ft) at the street-wall end away from the front door, on a main
      block whose street wall is >= 26 ft."""
    from shapely.geometry import Point
    defaults = []
    for d in doors:
        if d[5] in ('inferred-street', 'inferred-street-paving', 'mapped', 'inferred-path') and len(d) == 6:
            d.append('stoop'); defaults.append(dict(feature='stoop', status='default', method='typical for a detached house'))
    if any(d[5] == 'inferred-driveway' for d in doors): return doors, defaults
    main = max(R['parts'], key=lambda p_: p_['poly'].area)['poly']
    co = list(main.exterior.coords); best = None
    for (x0, z0), (x1, z1) in zip(co[:-1], co[1:]):
        L = math.hypot(x1 - x0, z1 - z0)
        if L < 26: continue
        tx, tz = (x1 - x0) / L, (z1 - z0) / L; nx, nz = tz, -tx
        if main.contains(Point((x0 + x1) / 2 + nx * 0.5, (z0 + z1) / 2 + nz * 0.5)): nx, nz = -nx, -nz
        f = nx * front[0] + nz * front[1]
        if best is None or f > best[0]: best = (f, x0, z0, tx, tz, nx, nz, L)
    if best is None or best[0] < 0.5: return doors, defaults
    f, x0, z0, tx, tz, nx, nz, L = best
    fd = next((d for d in doors if d[5] != 'inferred-driveway'), None)
    tdoor = ((fd[0] - x0) * tx + (fd[2] - z0) * tz) if fd else L * 0.3
    n = 2 if L >= 44 else 1; gw = 8.6; zone = n * (gw + 1.2)
    c = (L - 2.0 - zone / 2) if tdoor < L / 2 else (2.0 + zone / 2)
    if fd is not None and abs(c - tdoor) < zone / 2 + 3: return doors, defaults
    for k in range(n):
        t = c + (k - (n - 1) / 2) * (gw + 1.2)
        px, pz = x0 + tx * t, z0 + tz * t
        doors.append([round(px, 2), round(float(terr(px + nx * 2, pz + nz * 2)), 2), round(pz, 2), round(math.atan2(nx, nz), 4), gw, 'default-garage'])
    defaults.append(dict(feature='garage door', count=n, status='default', method='typical for a detached house whose street wall is >= 26 ft; replaced by driveway evidence when found'))
    return doors, defaults


def mesh_building(pg, Pp, terr, b, tags, cut_lines, ways, nodes, roads_xy, G, GEO, drive_xy=None, street_src='mapped', front=(0.0, 1.0), context=None):
    """Any building that is not a simple house: explicit parts with fitted roof primitives (solids.py), one
    consistent exterior, openings on exposed walls, plain doors labelled with their evidence (C1-C3)."""
    import solids as SO
    import shapely
    seed = abs(hash(tuple(np.round(pg.centroid.coords[0], 1)))) % 100000
    btype = classify_type(tags, pg, Pp, terr, context)
    R = SO.build(pg, Pp, terr, seed=seed, cut_lines=cut_lines, btype=btype, front=tuple(front))
    if R.get('template') and R['template']['status'] == 'chosen':
        # the template regularised the walls: doors, windows and the outline follow the built footprint
        from shapely.ops import unary_union
        fin = unary_union([p_['poly'] for p_ in R['parts']]).buffer(0)
        fin = max(GEO.as_polys(fin), key=lambda q: q.area, default=None)
        if fin is not None: pg = fin
    grade = float(np.median(terr(*np.array(pg.exterior.coords).T)))
    big = pg.area > 1800
    use = tags.get('building') or ('house' if pg.area < 2500 else 'yes')
    doors = GEO.entrances(pg, ways, nodes, roads_xy, terr, big, drive_xy=drive_xy, garage_ok=pg.area < 4000, street_src=street_src)
    defaults = []
    if btype['type'] == 'house' and R.get('template') and R['template']['status'] == 'chosen':
        doors, defaults = typed_house_defaults(R, doors, front, terr)
    wins, fh, base = SO.openings(pg, R, terr, doors, use)
    # summary heights from the reconstructed parts (area-weighted)
    tops, eaves_, areas = [], [], []
    for prt in R['parts']:
        c_ = prt['poly'].representative_point()
        x0_, z0_, x1_, z1_ = prt['poly'].bounds
        gx, gz = np.meshgrid(np.linspace(x0_, x1_, 12), np.linspace(z0_, z1_, 12))
        ins = shapely.contains_xy(prt['poly'], gx, gz)
        hv = prt['fit'].h(gx[ins], gz[ins]) if ins.any() else prt['fit'].h(np.array([c_.x]), np.array([c_.y]))
        tops.append(float(np.max(hv))); eaves_.append(float(np.min(hv))); areas.append(prt['poly'].area)
    top = max(tops) - grade
    eave = float(np.average(eaves_, weights=areas)) - grade
    kinds = sorted({p_['stats']['kind'] for p_ in R['parts']})
    ho = [p_['stats']['heldout'] for p_ in R['parts'] if p_['stats']['heldout']['n']]
    parts_ev = []
    for i, p_ in enumerate(R['parts']):
        st = p_['stats']
        parts_ev.append(dict(id=f'p{i}', kind=st['kind'], area_ft2=st.get('area_ft2'),
                             status='measured' if st['kind'] not in ('envelope', 'insufficient') else 'default',
                             method={'envelope': 'envelope: none of the implemented roof shapes passed the residual gate',
                                     'planar-brep': 'piecewise-planar roof: one supported plane, joints on plane intersections'}.get(st['kind'], 'roof primitive selection (penalised robust fit)'),
                             reason=st.get('reason'),
                             support=dict(n=st.get('n'), coverage=st.get('coverage'), density=st.get('density'),
                                          residual=dict(stat='median_abs', value=st.get('median_abs'), units='ft') if st.get('median_abs') is not None else None),
                             heldout=st['heldout'], alternatives=st.get('alternatives', []),
                             shape=st.get('shape'), plausibility=st.get('plausibility')))
    cen = pg.centroid
    pitched = [p_ for p_ in R['parts'] if p_['stats']['kind'] in ('gable', 'hip', 'hipped', 'shed', 'dome', 'planar-brep')]
    return dict(type='mesh', mesh_v=3, x=round(cen.x, 2), z=round(cen.y, 2), rot=0.0, grade=round(grade, 2),
                w=round(pg.bounds[2] - pg.bounds[0], 1), d=round(pg.bounds[3] - pg.bounds[1], 1),
                wall_h=round(eave, 1), eave_h=round(eave, 1), ridge_h=round(top, 1), area_ft2=round(pg.area),
                stories=max(1, int(round(max(eave, 8) / fh))), floor_h=fh,
                roof=dict(type='parts', kinds=kinds, parts=len(R['parts']), planes=len(R['parts']),
                          pitch_deg=0.0 if not pitched else None),
                mesh=R['mesh'], windows=wins, doors=doors, parts=parts_ev,
                _heldout=R['heldout'], building_type=btype, template=R.get('template'), defaults=defaults,
                footprint=[[round(x_, 1), round(z_, 1)] for x_, z_ in pg.exterior.coords[:-1]],
                validation=dict(heldout_points=R['holdout_n'],
                                seam_reconciliation=R.get('snap'),
                                heldout_parts=len(ho), underpowered_parts=sum(1 for h in ho if h['underpowered']) + sum(1 for p_ in R['parts'] if not p_['stats']['heldout']['n'])),
                provenance=dict(heights='measured', roof='measured parts' if 'envelope' not in kinds else 'measured parts; some envelopes',
                                windows='generated', doors=sorted({d[5] for d in doors})))

def morphology_label_keep_large(M, G, min_ft2):
    from scipy import ndimage
    L, n = ndimage.label(M)
    if n == 0: return M
    sizes = ndimage.sum(M, L, range(1, n + 1)) * G.c ** 2
    keep = np.isin(L, [k + 1 for k, s_ in enumerate(sizes) if s_ >= min_ft2])
    return keep


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
    analyze(ap.parse_args().folder)
