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

FT = 3.28084
NONBUILD = {2, 7, 9, 17, 18}          # ground, noise, water, bridge deck, high noise
rng = np.random.default_rng(7)


# ---------------------------------------------------------------- inputs
def load_site(folder):
    site = json.load(open(os.path.join(folder, 'site.json')))
    if site.get('status') != 'complete':
        sys.exit(f"site.json status is {site.get('status')!r}, expected 'complete'")
    dem = json.load(open(os.path.join(folder, 'dem.json')))
    n, c = dem['rows'], dem['cell_size_ft']
    E = np.array(dem['elevations_ft'], float).reshape(n, n)
    roads = None
    p = os.path.join(folder, 'roads.json')
    if os.path.exists(p):
        r = json.load(open(p))
        roads = r.get('roads', []) if isinstance(r, dict) else r
    return site, E, c, roads


class Terrain:
    def __init__(self, E, cell):
        self.E, self.c, self.n = E, cell, E.shape[0]
        self.pad = 0.0
        self.pad = float(self(0.0, 0.0))

    def __call__(self, x, z):
        x = np.asarray(x, float); z = np.asarray(z, float)
        fx = np.clip(x / self.c + self.n / 2 - .5, 0, self.n - 1.001)
        fz = np.clip(z / self.c + self.n / 2 - .5, 0, self.n - 1.001)
        j = np.floor(fx).astype(int); i = np.floor(fz).astype(int); u = fx - j; v = fz - i
        E = self.E
        h = E[i, j] * (1 - u) * (1 - v) + E[i, j + 1] * u * (1 - v) + E[i + 1, j] * (1 - u) * v + E[i + 1, j + 1] * u * v
        return h - self.pad


def dataset_year(name):
    m = re.search(r'(20\d\d|19\d\d)', name)
    if m: return int(m.group(1))
    m = re.search(r'_D(\d\d)\b', name)
    return 2000 + int(m.group(1)) if m else 0


def load_lidar(folder, site, terrain, radius=330):
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
            m = (np.abs(x) < radius) & (np.abs(z) < radius)
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


def pavement(prim, extent, n=240):
    """Paved ground (asphalt) from LiDAR return intensity: asphalt is dark at 1064 nm, grass is bright.
    Returns a run-length-encoded n x n mask over the terrain square, rows north to south."""
    from scipy import ndimage
    from skimage import filters, morphology
    g = prim['cls'] == 2
    if g.sum() < 500: return None
    x, z, it = prim['x'][g], prim['z'][g], prim['inten'][g].astype(float)
    cell = extent / n
    j = np.floor(x / cell + n / 2).astype(int); i = np.floor(z / cell + n / 2).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < n) & (j < n)
    s_ = np.zeros((n, n)); c_ = np.zeros((n, n))
    np.add.at(s_, (i[ok], j[ok]), np.log1p(it[ok])); np.add.at(c_, (i[ok], j[ok]), 1)
    has = c_ > 0
    if has.sum() < 200: return None
    v = np.where(has, s_ / np.maximum(c_, 1), np.nan)
    thr = filters.threshold_otsu(v[has])
    lo, hi = np.percentile(v[has], [10, 90])
    if hi - lo < 0.6: return None                        # no usable intensity contrast
    dark = has & (v < thr)
    # ground seen through tree canopy is shaded and dark too: drop cells under vegetation
    veg = (~np.isin(prim['cls'], [2, 7, 9, 17, 18])) & (prim['nret'] > 1) & (prim['hag'] > 3)
    vj = np.floor(prim['x'][veg] / cell + n / 2).astype(int); vi = np.floor(prim['z'][veg] / cell + n / 2).astype(int)
    vk = (vi >= 0) & (vj >= 0) & (vi < n) & (vj < n)
    canopy = np.zeros((n, n), bool); canopy[vi[vk], vj[vk]] = True
    canopy = ndimage.binary_dilation(canopy, iterations=1)
    dark &= ~canopy
    dark = ndimage.median_filter(dark.astype(np.uint8), size=3).astype(bool)
    dark = morphology.remove_small_objects(dark, max_size=40)
    dark = morphology.remove_small_holes(dark, max_size=30)
    rle = []
    for row in dark:
        runs = []; k = 0
        while k < n:
            if row[k]:
                s0 = k
                while k < n and row[k]: k += 1
                runs += [s0, k - s0]
            else: k += 1
        rle.append(runs)
    return dict(n=n, extent=round(extent, 3), rle=rle, share=round(float(dark.mean()), 3))


class Grid:
    """The terrain square as an n x n raster (rows north to south), for the site zone and pavement."""
    def __init__(self, extent, n=240):
        self.n, self.extent, self.c = n, extent, extent / n

    def ij(self, x, z):
        j = np.floor(np.asarray(x) / self.c + self.n / 2).astype(int); i = np.floor(np.asarray(z) / self.c + self.n / 2).astype(int)
        return i, j, (i >= 0) & (j >= 0) & (i < self.n) & (j < self.n)

    def raster(self, x, z):
        i, j, ok = self.ij(x, z); M = np.zeros((self.n, self.n), bool); M[i[ok], j[ok]] = True; return M

    def lookup(self, M, x, z):
        i, j, ok = self.ij(x, z); out = np.zeros(len(np.atleast_1d(x)), bool)
        out[ok] = M[i[ok], j[ok]]; return out

    def xz(self, i, j):
        return (np.asarray(j) + .5 - self.n / 2) * self.c, (np.asarray(i) + .5 - self.n / 2) * self.c


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


def analyze(folder, verbose=True, buffer_ft=100.0, whole_max_ft2=40000.0):
    """Site rule: the building under the address point, plus every building and ground feature that
    touches the zone within buffer_ft of its footprint. Touching buildings are kept whole up to
    whole_max_ft2 of roof, larger ones are clipped to the zone."""
    from scipy import ndimage
    site, E, cell, roads = load_site(folder)
    terr = Terrain(E, cell)
    D = load_lidar(folder, site, terr)
    prim = D[0]
    extent = E.shape[0] * cell; G = Grid(extent); half = extent / 2 - 3
    X = np.concatenate([d['x'] for d in D]); Y = np.concatenate([d['y'] for d in D]); Z = np.concatenate([d['z'] for d in D])
    C = np.concatenate([d['cls'] for d in D]); H = np.concatenate([d['hag'] for d in D])
    NR = np.concatenate([d['nret'] for d in D])
    S = np.concatenate([np.full(d['n'], i) for i, d in enumerate(D)])
    pave = pavement(prim, extent)
    # roofs return one echo; tree canopy returns several. Keep single returns for building detection.
    m = (~np.isin(C, list(NONBUILD))) & (H > 6) & (H < 260) & (np.abs(X) < half) & (np.abs(Z) < half) & (NR <= 1)
    x, y, z = X[m], Y[m], Z[m]
    lab, n = clusters(x, z, cell=3.0, dilate=0)
    polys = road_polylines(roads)
    cands = []
    for k in range(1, n + 1):
        I = np.where(lab == k)[0]
        if len(I) < 60: continue
        P = np.c_[x[I], y[I], z[I]]
        planes = ransac_planes(P)
        roof = [(cf, J) for cf, J in planes if plane_info(cf)['pitch'] < 55]
        nroof = sum(len(J) for _, J in roof)
        if not roof or nroof < 50: continue
        J = np.concatenate([J for _, J in roof])
        R = min_rect(P[J, 0], P[J, 2])
        frac = nroof / len(I); big = max(len(J) for _, J in roof)
        if R['area'] < 350 or big < 35: continue            # trees: many small noisy planes
        if frac < 0.35 and big / len(I) < 0.2: continue
        cands.append(dict(I=I, P=P, planes=planes, rect=R, n=len(I), d0=float(np.hypot(P[:, 0], P[:, 2]).min())))
    if not cands: sys.exit('no buildings found in the LiDAR near this address')
    feat = min(cands, key=lambda c: c['d0'])
    if feat['d0'] > 120:
        sys.exit(f'no building within 120 ft of the address point (nearest roof is {feat["d0"]:.0f} ft away); '
                 'the geocode may be off, or the LiDAR predates the building')
    # footprints on the site grid
    def footprint(c):
        M = G.raster(c['P'][:, 0], c['P'][:, 2])
        M = ndimage.binary_closing(M, iterations=2) | M
        return ndimage.binary_fill_holes(M)
    for c in cands: c['fp'] = footprint(c); c['area'] = float(c['fp'].sum() * G.c ** 2)
    zone = ndimage.distance_transform_edt(~feat['fp']) * G.c <= buffer_ft
    inc = [c for c in cands if (c['fp'] & zone).any()]
    # where is the street? OSM centreline if we have it, else the nearest LiDAR pavement
    pave_mask = None
    if pave:
        pave_mask = np.zeros((pave['n'], pave['n']), bool)
        for i_, runs in enumerate(pave['rle']):
            for k in range(0, len(runs), 2): pave_mask[i_, runs[k]:runs[k] + runs[k + 1]] = True
        big_p = morphology_label_keep_large(pave_mask, G, min_ft2=1500)
        pidx = ndimage.distance_transform_edt(~big_p, return_distances=False, return_indices=True) if big_p.any() else None
    def street_dir(pt):
        if polys:
            q = nearest_road(np.array(pt), polys)
            if q: return (q[1] - pt) / (np.hypot(*(q[1] - pt)) or 1), q[3].get('name') if q[3] else None
        if pave_mask is not None and pidx is not None:
            i, j, ok = G.ij([pt[0]], [pt[1]])
            if ok[0]:
                pi, pj = pidx[0][i[0], j[0]], pidx[1][i[0], j[0]]
                qx, qz = G.xz(pi, pj); v = np.array([qx - pt[0], qz - pt[1]])
                if np.hypot(*v) > 1: return v / np.hypot(*v), None
        return np.array([0.0, 1.0]), None
    fpt = np.array([feat['rect']['cx'], feat['rect']['cz']])
    front, street_name = street_dir(fpt)
    blds = []
    for c in inc:
        is_feat = c is feat
        clip = (not is_feat) and c['area'] > whole_max_ft2
        cpt = np.array([c['rect']['cx'], c['rect']['cz']])
        hh = None
        if not clip and c['area'] <= 3500:
            fv, _ = street_dir(cpt)
            res = measure_building(c['P'], c['planes'], terr, fv)
            hh = res[0] if res else None
            if not house_like(c, hh): hh = None
        if hh is None:
            zf = (lambda xx, zz: G.lookup(zone, xx, zz)) if clip else None
            hh = roofgrid(c, terr, cell=2.0 if c['area'] < 12000 else 2.5, zone=zf)
            hh['clipped'] = bool(clip)
        hh.update(id='featured' if is_feat else f'b{len(blds)}', role='featured' if is_feat else 'context')
        blds.append(hh)
    blds.sort(key=lambda h: h['role'] != 'featured')
    # final site = zone plus the whole footprints of included buildings (with a little yard)
    site_m = zone.copy()
    for c in inc:
        if c['area'] <= whole_max_ft2 or c is feat: site_m |= c['fp']
    site_m = ndimage.binary_dilation(site_m, iterations=2)
    site_m = ndimage.binary_fill_holes(ndimage.binary_closing(site_m, iterations=3))
    # trees: canopy peaks (primary dataset) inside the site, not on included roofs
    roofs = np.zeros_like(site_m)
    for c in inc: roofs |= ndimage.binary_dilation(c['fp'], iterations=2)
    tm = (S == 0) & (~np.isin(C, list(NONBUILD))) & (H > 12) & (H < 140) & (NR > 1)
    tm &= G.lookup(site_m, X, Z) & ~G.lookup(roofs, X, Z)
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
            if len(trees) >= 500: break
    # featured outline (for the yellow line on the ground)
    from skimage import measure
    outline = []
    fo = np.pad(ndimage.binary_dilation(feat['fp'], iterations=2), 1).astype(float)
    cs = measure.find_contours(fo, 0.5)
    if cs:
        cnt = measure.approximate_polygon(max(cs, key=len), tolerance=0.8)
        ox, oz = G.xz(cnt[:, 0] - 1 - .5 + .5, cnt[:, 1] - 1 - .5 + .5)
        outline = [[round(float(a_), 1), round(float(b_), 1)] for a_, b_ in zip(ox, oz)]
    scene = dict(schema_version=2, address=site['address'], matched_address=site.get('matched_address'),
                 lat=site['lat'], lon=site['lon'], pad_ft=round(terr.pad, 2), buffer_ft=buffer_ft,
                 datasets=[dict(name=d['name'], year=d['year'], points=int(d['n']), z_offset_ft=round(d['offset_ft'], 2),
                                fit_spread_ft=round(d['spread_ft'], 2)) for d in D],
                 street=dict(front=[round(float(front[0]), 4), round(float(front[1]), 4)], name=street_name,
                             source='osm' if polys else ('pavement' if pave else 'none')),
                 site=dict(n=G.n, extent=round(extent, 3), rle=rle(site_m)),
                 outline=outline, houses=blds, trees=trees, pavement=pave, candidates=len(cands))
    json.dump(scene, open(os.path.join(folder, 'scene.json'), 'w'), separators=(',', ':'))
    if verbose:
        for hh in blds:
            r = hh['roof']
            desc = (f"grid {r['planes']} planes, pitch {r['pitch_deg']}, {hh['area_ft2']} ft2{' clipped' if hh.get('clipped') else ''}"
                    if r['type'] == 'grid' else f"{r['type']} pitch {r.get('pitch_deg')} cross {len(r.get('cross', []))}")
            print(f"{hh['id']:9s} ({hh['x']:7.1f},{hh['z']:7.1f}) {hh['w']}x{hh['d']}  eave {hh['wall_h']}  top {hh['ridge_h']}  {desc}")
        print(f"buildings {len(blds)}, trees {len(trees)}, street {scene['street']['source']} {street_name or ''}, "
              f"site {site_m.mean() * extent ** 2:.0f} ft2, pavement {pave['share'] if pave else 'none'}")
    return scene


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
