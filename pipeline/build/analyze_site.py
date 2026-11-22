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
                                  np.asarray(l.number_of_returns)[m]])
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
                        nret=P[:, 4].astype(int), hag=y - T, offset_ft=off, spread_ft=spread, n=len(P)))
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


def massing(c, terr, front_vec, cell=3.0):
    """Large or complex building: its measured shape as columns on a 3 ft grid (top = 95th pct height)."""
    from scipy import ndimage
    P = c['P']
    x0, z0 = float(P[:, 0].min()) - cell, float(P[:, 2].min()) - cell
    j = ((P[:, 0] - x0) / cell).astype(int); i = ((P[:, 2] - z0) / cell).astype(int)
    G = np.full((i.max() + 2, j.max() + 2), np.nan)
    order = np.lexsort((P[:, 1], i, j))
    for a, b, yv in zip(i[order], j[order], P[order, 1]):
        G[a, b] = yv if np.isnan(G[a, b]) else max(G[a, b], yv)
    occ = ~np.isnan(G)
    filled = ndimage.binary_closing(occ, iterations=1) | occ
    filled = ndimage.binary_fill_holes(filled)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = ndimage.generic_filter(np.where(occ, G, np.nan), np.nanmedian, size=3, mode='constant', cval=np.nan)
    top = np.where(np.isnan(med), G, med)                       # 3x3 median smooths edge hits and rooftop clutter
    top = np.where(np.isnan(top), np.nanmedian(G), top)
    top = np.round(top * 2) / 2
    cells = []
    for a, b in zip(*np.nonzero(filled)):
        cells.append([int(a), int(b), int(round(float(top[a, b]) * 10))])
    xs = x0 + (np.nonzero(filled)[1] + .5) * cell; zs = z0 + (np.nonzero(filled)[0] + .5) * cell
    cx, cz = float(xs.mean()), float(zs.mean())
    tops = np.array([t for _, _, t in cells]) / 10
    grade = float(np.median(terr(xs, zs)))
    rot = math.atan2(front_vec[0], front_vec[1])
    area = len(cells) * cell * cell
    return dict(type='massing', x=round(cx, 2), z=round(cz, 2), rot=round(rot, 4), grade=round(grade, 2),
                w=round(float(xs.max() - xs.min() + cell), 1), d=round(float(zs.max() - zs.min() + cell), 1),
                wall_h=round(float(np.median(tops) - grade), 1), ridge_h=round(float(tops.max() - grade), 1),
                eave_h=round(float(np.median(tops) - grade), 1), area_ft2=round(area),
                stories=max(1, int(round((np.median(tops) - grade) / 12))),
                roof=dict(type='massing', pitch_deg=0),
                grid=dict(x0=round(x0, 2), z0=round(z0, 2), cell=cell, cells=cells))


def analyze(folder, verbose=True):
    site, E, cell, roads = load_site(folder)
    terr = Terrain(E, cell)
    D = load_lidar(folder, site, terr)
    prim = D[0]
    X = np.concatenate([d['x'] for d in D]); Y = np.concatenate([d['y'] for d in D]); Z = np.concatenate([d['z'] for d in D])
    C = np.concatenate([d['cls'] for d in D]); H = np.concatenate([d['hag'] for d in D])
    NR = np.concatenate([d['nret'] for d in D])
    S = np.concatenate([np.full(d['n'], i) for i, d in enumerate(D)])
    # roofs return one echo; tree canopy returns several. Keep single returns for building detection.
    m = (~np.isin(C, list(NONBUILD))) & (H > 6) & (H < 260) & (np.abs(X) < 260) & (np.abs(Z) < 260) & (NR <= 1)
    x, y, z, h, s = X[m], Y[m], Z[m], H[m], S[m]
    lab, n = clusters(x, z, cell=3.0, dilate=0)
    polys = road_polylines(roads)
    cands = []
    for k in range(1, n + 1):
        I = np.where(lab == k)[0]
        if len(I) < 60: continue
        P = np.c_[x[I], y[I], z[I]]
        planes = ransac_planes(P)
        roof = [(c, J) for c, J in planes if plane_info(c)['pitch'] < 55]
        nroof = sum(len(J) for _, J in roof)
        if not roof or nroof < 50: continue
        J = np.concatenate([J for _, J in roof])
        R = min_rect(P[J, 0], P[J, 2])
        frac = nroof / len(I)
        big = max(len(J) for _, J in roof)
        # trees give many small noisy planes; roofs give a few large ones
        if R['area'] < 350 or big < 35: continue
        if frac < 0.35 and big / len(I) < 0.2: continue
        cands.append(dict(I=I, P=P, planes=planes, rect=R, frac=frac, n=len(I), large=R['area'] > 15000,
                          d0=float(np.hypot(P[:, 0], P[:, 2]).min())))
    if not cands: sys.exit('no buildings found in the LiDAR near this address')
    # featured: building containing / nearest the geocode
    # the building the address point falls on (or the nearest one's closest roof point)
    feat = min(cands, key=lambda c: c['d0'])
    if feat['d0'] > 120:
        sys.exit(f'no building within 120 ft of the address point (nearest roof is {feat["d0"]:.0f} ft away); '
                 'the geocode may be off, or the LiDAR predates the building')
    fpt = np.array([feat['rect']['cx'], feat['rect']['cz']])
    nr = nearest_road(fpt, polys) if polys else None
    if nr:
        front = (nr[1] - fpt) / (np.hypot(*(nr[1] - fpt)) or 1); tangent = nr[2]
    else:
        # fallback: houses line up along the street, so take the street direction from the row of
        # nearby buildings, then pick the side with less vegetation (front yards + street are open)
        cen = np.array([[c['rect']['cx'], c['rect']['cz']] for c in cands])
        dd = np.hypot(*(cen - fpt).T); near = cen[np.argsort(dd)[:5]]
        if len(near) >= 3:
            q = near - near.mean(0); tang = np.linalg.svd(q, full_matrices=False)[2][0]
        else:
            tang = np.array([math.cos(feat['rect']['ang']), math.sin(feat['rect']['ang'])])
        nrm = np.array([-tang[1], tang[0]])
        veg = (~np.isin(prim['cls'], [2, 7, 9, 17, 18])) & (prim['hag'] > 3) & (prim['nret'] > 1)   # canopy, not roofs
        q = np.c_[prim['x'][veg] - fpt[0], prim['z'][veg] - fpt[1]]
        best = None
        for sgn in (1, -1):
            dv = nrm * sgn; along = q @ dv; across = q @ tang
            sc = int(((along > 30) & (along < 90) & (np.abs(across) < 60)).sum())
            if best is None or sc < best[0]: best = (sc, dv)
        front = best[1]; tangent = tang

    def mk(c, front_vec):
        res = None if c['large'] else measure_building(c['P'], c['planes'], terr, front_vec)
        if res:
            hh = res[0]
            # a house-style fit that is out of house range falls back to massing
            if 8 <= hh['wall_h'] <= 40 and hh['ridge_h'] <= 60 and max(hh['w'], hh['d']) <= 130: return hh
        return massing(c, terr, front_vec)
    houses = []
    fh = mk(feat, front)
    if not fh: sys.exit('could not fit a roof to the featured building')
    fh.update(id='featured', role='featured'); houses.append(fh)
    # neighbours: same side of the street, nearest along the street on each side
    side0 = None
    if nr: side0 = np.sign((fpt - nr[1]) @ front * -1)
    picks = {}
    for c in cands:
        if c is feat: continue
        p = np.array([c['rect']['cx'], c['rect']['cz']]); dv = p - fpt
        dist = float(np.hypot(*dv))
        if dist < 25 or dist > 260: continue
        t = float(dv @ tangent); off = float(dv @ front)
        if abs(off) > max(45, abs(t) * 0.8): continue      # across the street or behind
        if nr:
            q = nearest_road(p, polys)
            if q and q[0] > 1.8 * nr[0] + 40: continue
        key = 'right' if t > 0 else 'left'
        if key not in picks or dist < picks[key][0]: picks[key] = (dist, c)
    for key, (dist, c) in picks.items():
        p = np.array([c['rect']['cx'], c['rect']['cz']])
        q = nearest_road(p, polys) if polys else None
        fv = (q[1] - p) / (np.hypot(*(q[1] - p)) or 1) if q else front
        hh = mk(c, fv)
        if hh: hh.update(id=key, role='neighbor'); houses.append(hh)
    # trees: primary dataset, non-building points, local maxima on a 10 ft grid
    trees = []
    tm = (S == 0) & (~np.isin(C, list(NONBUILD))) & (H > 12) & (H < 110) & (np.abs(X) < 320) & (np.abs(Z) < 320)
    for hh in houses + [dict(x=c['rect']['cx'], z=c['rect']['cz'], w=c['rect']['lu'], d=c['rect']['lv']) for c in cands]:
        r = max(hh['w'], hh['d']) / 2 + 3
        tm &= (X - hh['x']) ** 2 + (Z - hh['z']) ** 2 > r * r
    if tm.any():
        tx, tz, th = X[tm], Z[tm], H[tm]
        gi = np.floor(tx / 10).astype(int); gj = np.floor(tz / 10).astype(int)
        best = {}
        for a, b, hh, xx, zz in zip(gi, gj, th, tx, tz):
            k = (a, b)
            if k not in best or hh > best[k][0]: best[k] = (hh, xx, zz, 1 + (best.get(k, (0, 0, 0, 0))[3]))
            else: best[k] = (best[k][0], best[k][1], best[k][2], best[k][3] + 1)
        cellv = sorted(best.values(), key=lambda v: -v[0])
        for hh, xx, zz, cnt in cellv:
            if cnt < 3: continue
            if any((xx - t['x']) ** 2 + (zz - t['z']) ** 2 < 14 ** 2 for t in trees): continue
            trees.append(dict(x=round(float(xx), 1), z=round(float(zz), 1), h=round(float(min(hh, 95)), 1)))
            if len(trees) >= 450: break
    scene = dict(schema_version=1, address=site['address'], matched_address=site.get('matched_address'),
                 lat=site['lat'], lon=site['lon'], pad_ft=round(terr.pad, 2),
                 datasets=[dict(name=d['name'], year=d['year'], points=int(d['n']), z_offset_ft=round(d['offset_ft'], 2),
                                fit_spread_ft=round(d['spread_ft'], 2)) for d in D],
                 street=dict(front=[round(float(front[0]), 4), round(float(front[1]), 4)], tangent=[round(float(tangent[0]), 4), round(float(tangent[1]), 4)],
                             name=(nr[3].get('name') if nr else None), source='osm' if nr else 'inferred'),
                 houses=houses, trees=trees, candidates=len(cands))
    json.dump(scene, open(os.path.join(folder, 'scene.json'), 'w'), indent=1)
    if verbose:
        for hh in houses:
            print(f"{hh['id']:9s} at ({hh['x']:7.1f},{hh['z']:7.1f}) rot {hh['rot']:+.3f}  {hh['w']}x{hh['d']} ft  wall {hh['wall_h']}  ridge {hh['ridge_h']}  roof {hh['roof'] if hh['roof']['type'] != 'massing' else 'massing, ' + str(hh['area_ft2']) + ' ft2, ' + str(len(hh['grid']['cells'])) + ' columns'}")
        print(f"trees {len(trees)}, building candidates {len(cands)}, street source {scene['street']['source']}")
    return scene


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
    analyze(ap.parse_args().folder)
