"""
gates.py - acceptance gates on the EXPORTED (quantized) building meshes (contract C4, revised after review R2).

Every gate returns an explicit verdict: pass | fail | incomplete | not_applicable, with the population it checked.
The acceptance result aggregates every required gate: any fail -> fail; else any incomplete -> incomplete; an empty
population is not_applicable and never certifies anything. Topology-only is reported separately.

Per part-built building (decoded exactly as the page decodes it, welded only at identical int32 0.01 ft coordinates):
- topology: an empty mesh fails; degenerate triangles, non-manifold edges, edge-orientation conflicts,
  T-junctions (all open edges including the declared base, no truncation), open edges outside the declared base.
- base: the declared base loop must be closed, consist of bottom edges of vertical wall triangles, and lie at
  terrain - 1.5 ft (within 0.05 ft) when dem.json is available; otherwise the gate is incomplete. It follows the
  terrain, so it need not be horizontal on slopes.
- orientation: every connected component, capped at its base loops, must have positive signed volume (outward
  normals); a negative volume is an inverted shell.
- intersections: improper triangle crossings (beyond shared vertices/edges) and coplanar faces overlapping with
  positive area, whether or not they share vertices.
- held-out: withheld roof points (sidecar heldout.json written by the build) are measured against the EXPORTED roof
  triangles by vertical ray casting; any multiple hit fails (roofs here are single-valued in plan); missing hits over
  5% fail; spatial blocks and both signed tails are reported. < 50 points or < 3 blocks = incomplete (never pass). Thresholds are provisional development gates.
Buildings that are not part-built (simple parametric houses) are reported as not_checked, which keeps the site's
acceptance incomplete until that path exports checkable geometry.

Usage: python gates.py <site folder>  -> writes validation.json; exit 0 only if acceptance == pass.
"""
import base64, json, math, os, sys
from collections import defaultdict
import numpy as np

GATE_MEDIAN, GATE_P90, MIN_HELDOUT, MIN_BLOCKS = 0.8, 2.5, 50, 3
BASE_DROP, BASE_TOL = 1.5, 0.05
LAYERS = ('roof', 'roof_flat', 'walls', 'section')


def decode(m):
    if not m or not m.get('n'): return np.zeros((0, 3), np.int64)
    q = m.get('q')
    if q:
        a = np.frombuffer(base64.b64decode(m['p']), '<i4').astype(np.int64)
        if q != 100: a = np.round(a * (100.0 / q)).astype(np.int64)
    else:
        a = np.round(np.frombuffer(base64.b64decode(m['p']), '<i2').astype(np.float64) * 5).astype(np.int64)   # 0.05 ft legacy
    return a.reshape(-1, 3)


def verdict(*vs):
    vs = [v for v in vs if v != 'not_applicable']
    if not vs: return 'not_applicable'
    if 'fail' in vs: return 'fail'
    if 'incomplete' in vs: return 'incomplete'
    return 'pass'


# ------------------------------------------------------------------ geometry helpers (all in int 0.01 ft units)
def _seg_tri(p, q, a, b, c, eps=1e-9):
    """Proper crossing of segment pq with the interior of triangle abc (touching within eps is not a crossing)."""
    e1, e2 = b - a, c - a
    d = q - p
    h = np.cross(d, e2); det = e1 @ h
    if abs(det) < eps: return False
    f = 1.0 / det; s = p - a
    u = f * (s @ h)
    if u <= 1e-7 or u >= 1 - 1e-7: return False
    qv = np.cross(s, e1); v = f * (d @ qv)
    if v <= 1e-7 or u + v >= 1 - 1e-7: return False
    t = f * (e2 @ qv)
    return 1e-7 < t < 1 - 1e-7


def intersections(V, T):
    """Improper crossings between triangles that do not share an edge (pairs sharing a vertex are tested on their
    non-shared edges) + coplanar overlapping duplicates. Broad phase: uniform 3D grid on triangle bounding boxes."""
    Vf = V.astype(float)
    tri = Vf[T]
    lo, hi = tri.min(1), tri.max(1)
    cell = max(200.0, float(np.median(hi - lo)) * 2)                   # 2 ft minimum
    grid = defaultdict(list)
    for i in range(len(T)):
        a0 = np.floor(lo[i] / cell).astype(int); a1 = np.floor(hi[i] / cell).astype(int)
        for gx in range(a0[0], a1[0] + 1):
            for gy in range(a0[1], a1[1] + 1):
                for gz in range(a0[2], a1[2] + 1):
                    grid[(gx, gy, gz)].append(i)
    seen = set(); crossing = duplicate = 0; examples = []
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for ids in grid.values():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                key = (i, j) if i < j else (j, i)
                if key in seen: continue
                seen.add(key)
                if np.any(lo[i] > hi[j]) or np.any(lo[j] > hi[i]): continue
                shared = set(T[i].tolist()) & set(T[j].tolist())
                ni, nj = nrm[i], nrm[j]
                nn = np.linalg.norm(ni) * np.linalg.norm(nj)
                coplanar = np.linalg.norm(np.cross(ni, nj)) < 1e-9 * nn + 1e-12 and abs(ni @ (tri[j][0] - tri[i][0])) < 1e-3 * np.linalg.norm(ni)
                if coplanar:
                    # coplanar faces may touch (shared vertex/edge) but must not overlap with positive area
                    from shapely.geometry import Polygon
                    ax = int(np.argmax(np.abs(ni))); keep = [d for d in range(3) if d != ax]
                    pa = Polygon(tri[i][:, keep]); pb = Polygon(tri[j][:, keep])
                    if pa.is_valid and pb.is_valid and pa.intersection(pb).area > 1.0:      # > 0.0001 ft2
                        duplicate += 1
                        if len(examples) < 10: examples.append(('coplanar-overlap', key))
                    continue
                if len(shared) >= 2: continue                   # non-coplanar faces sharing an edge: a legitimate fold
                hit = False
                for (A, B, ta) in ((tri[i], tri[j], T[i]), (tri[j], tri[i], T[j])):
                    for k in range(3):
                        u, w = ta[k], ta[(k + 1) % 3]
                        if u in shared and w in shared: continue
                        if _seg_tri(A[k], A[(k + 1) % 3], B[0], B[1], B[2]): hit = True; break
                    if hit: break
                if hit:
                    crossing += 1
                    if len(examples) < 10: examples.append(('cross', key))
    return crossing, duplicate, examples


def components(T, nverts):
    parent = list(range(nverts))
    def f(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for t in T:
        a, b, c = (f(int(v)) for v in t)
        parent[b] = a; parent[f(c)] = a
    roots = defaultdict(list)
    for ti, t in enumerate(T): roots[f(int(t[0]))].append(ti)
    return list(roots.values())


def terrain_from(folder):
    p = os.path.join(folder, 'dem.json'); s = os.path.join(folder, 'scene.json')
    if not os.path.exists(p): return None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_site import Terrain, dem_bounds
    dem = json.load(open(p))
    E = np.array(dem['elevations_ft'], float).reshape(dem['rows'], dem.get('cols', dem['rows']))
    T = Terrain(E, dem['cell_size_ft'], dem_bounds(dem))
    try:
        pad = json.load(open(s)).get('pad_ft')
        if pad is not None: T.pad = float(pad)
    except Exception: pass
    return T


# ------------------------------------------------------------------ per building
def check_building(h, terr=None, heldout=None):
    M = h['mesh']
    P = np.concatenate([decode(M.get(k)) for k in LAYERS])
    layer_of = np.concatenate([np.full(len(decode(M.get(k))) // 3, i) for i, k in enumerate(LAYERS)]) if len(P) else np.zeros(0, int)
    V, inv = np.unique(P, axis=0, return_inverse=True)
    T = inv.reshape(-1, 3)
    Vf = V.astype(float)
    cr = np.cross(Vf[T[:, 1]] - Vf[T[:, 0]], Vf[T[:, 2]] - Vf[T[:, 0]])
    area2 = np.linalg.norm(cr, axis=1)
    degenerate = int((area2 < 1.0).sum())
    good = area2 >= 1.0
    T, layer_of = T[good], layer_of[good]
    edges = defaultdict(list)
    for ti, t in enumerate(T):
        for i in range(3):
            u, v = int(t[i]), int(t[(i + 1) % 3])
            edges[(min(u, v), max(u, v))].append((u, v, ti))
    kid = {tuple(k): i for i, k in enumerate(V.tolist())}
    declared = set()
    for a_, b_ in M.get('base') or []:
        qa = tuple(int(round(c * 100)) for c in a_); qb = tuple(int(round(c * 100)) for c in b_)
        if qa in kid and qb in kid and qa != qb: declared.add((min(kid[qa], kid[qb]), max(kid[qa], kid[qb])))
    open_edges = [e for e, us in edges.items() if len(us) == 1]
    nonmanifold = sum(1 for us in edges.values() if len(us) > 2)
    orient_conf = sum(1 for us in edges.values() if len(us) == 2 and us[0][:2] == us[1][:2])
    undeclared_open = [e for e in open_edges if e not in declared]
    declared_unused = [e for e in declared if e not in edges or len(edges[e]) != 1]
    # T-junctions: a vertex strictly inside an open edge (all of them, no truncation)
    tj = 0
    if open_edges:
        from scipy.spatial import cKDTree
        tree = cKDTree(Vf)
        for u, v in open_edges:
            p, q = Vf[u], Vf[v]; d = q - p; L = np.linalg.norm(d)
            if L < 1: continue
            for k in tree.query_ball_point((p + q) / 2, L / 2 + 1):
                if k in (u, v): continue
                t = (Vf[k] - p) @ d / (L * L)
                if 1e-4 < t < 1 - 1e-4 and np.linalg.norm(Vf[k] - (p + t * d)) < 1.0: tj += 1; break
    topo = dict(degenerate=degenerate, nonmanifold=nonmanifold, orientation_conflicts=orient_conf, t_junctions=tj,
                open_edges_not_declared=len(undeclared_open), declared_base_edges=len(declared),
                declared_but_not_open=len(declared_unused))
    empty = len(T) == 0
    topo['triangles_checked'] = int(len(T))
    topo['verdict'] = 'fail' if empty else 'pass' if (degenerate == 0 and nonmanifold == 0 and orient_conf == 0 and tj == 0 and not undeclared_open and not declared_unused) else 'fail'
    # base loops: closed, bottoms of vertical wall triangles, at terrain - 1.5
    base_v = 'not_applicable'; base_detail = {}
    if declared:
        deg = defaultdict(int)
        for u, v in declared: deg[u] += 1; deg[v] += 1
        not_closed = sum(1 for d in deg.values() if d != 2)
        not_wall_bottom = 0
        for (u, v) in declared:
            if (u, v) not in edges: not_wall_bottom += 1; continue
            ti = edges[(u, v)][0][2]; t = T[ti]
            w = [int(x) for x in t if x not in (u, v)][0]
            # the third vertex must be above the edge (a wall rising from it), and the triangle must be a wall
            if not (LAYERS[layer_of[ti]] in ('walls', 'section') and V[w][1] > min(V[u][1], V[v][1])): not_wall_bottom += 1
        off_terrain = None
        if terr is not None:
            vs = sorted(deg)
            dev = np.array([abs(Vf[i][1] / 100 - (float(terr(Vf[i][0] / 100, Vf[i][2] / 100)) - BASE_DROP)) for i in vs])
            off_terrain = int((dev > BASE_TOL + 0.011).sum())
        base_detail = dict(edges=len(declared), vertices_not_degree_2=not_closed, edges_not_wall_bottoms=not_wall_bottom,
                           vertices_off_terrain=off_terrain)
        if not_closed or not_wall_bottom or (off_terrain or 0): base_v = 'fail'
        elif off_terrain is None: base_v = 'incomplete'
        else: base_v = 'pass'
    # outward orientation per connected component, capped at its open boundary loops
    comps = components(T, len(V)) if len(T) else []
    inverted = 0; vols = []
    for cmp_ in comps:
        vol = 0.0; bverts = []
        for ti in cmp_:
            a, b, c = Vf[T[ti]]
            vol += a @ np.cross(b, c)
        cen = None
        bd = [(u, v, ti) for (e, us) in edges.items() if len(us) == 1 for (u, v, ti) in us if ti in set(cmp_)]
        if bd:
            cen = Vf[[x for u, v, _ in bd for x in (u, v)]].mean(0)
            for u, v, _ in bd:                     # cap: (centroid, v, u) closes the hole with consistent orientation
                vol += cen @ np.cross(Vf[v], Vf[u])
        vol /= 6.0
        vols.append(vol / 1e6)                     # ft3
        if vol < 0: inverted += 1
    tiny = sum(1 for v in vols if 0 <= v < 1.0)                 # a closed shell under 1 ft3 is not a building solid
    orient_v = 'fail' if (not comps or inverted or tiny) else 'pass'
    # intersections
    crossing, duplicate, ex = intersections(V, T) if len(T) else (0, 0, [])
    inter_v = 'fail' if not len(T) else ('fail' if (crossing or duplicate) else 'pass')
    # held-out on the exported roof
    ho_rows = []; ho_v = 'not_applicable'
    parts = h.get('parts') or []
    if parts:
        roofT = T[np.isin(layer_of, [0, 1])]
        if heldout is None:
            ho_v = 'incomplete'
            ho_rows = [dict(id=p['id'], verdict='incomplete', reason='no heldout.json sidecar') for p in parts]
        else:
            ho_rows = heldout_on_mesh(Vf / 100.0, roofT, heldout.get(h['id']) or {}, parts)
            ho_v = verdict(*[r['verdict'] for r in ho_rows]) if ho_rows else 'incomplete'
    acc = verdict(topo['verdict'], base_v, orient_v, inter_v, ho_v)
    return dict(id=h['id'], name=h.get('name'), triangles=int(len(T)), vertices=int(len(V)),
                topology=topo, base=dict(verdict=base_v, **base_detail),
                orientation=dict(verdict=orient_v, components=len(comps), inverted=inverted, below_1ft3=tiny, volumes_ft3=[round(v, 1) for v in vols[:20]]),
                intersections=dict(verdict=inter_v, crossings=crossing, duplicates=duplicate, examples=[list(map(str, e)) for e in ex[:5]]),
                heldout=dict(verdict=ho_v, parts=ho_rows),
                envelopes=sum(1 for p in parts if p['kind'] == 'envelope'), parts=len(parts),
                topology_only=topo['verdict'], acceptance=acc)


def heldout_on_mesh(V, roofT, pts_by_part, parts):
    """Vertical rays from withheld points to the exported roof triangles (2D point-in-triangle, grid index)."""
    tri = V[roofT]
    xz = tri[:, :, [0, 2]]
    lo, hi = xz.min(1), xz.max(1)
    cell = 10.0; grid = defaultdict(list)
    for i in range(len(tri)):
        for gx in range(int(math.floor(lo[i][0] / cell)), int(math.floor(hi[i][0] / cell)) + 1):
            for gz in range(int(math.floor(lo[i][1] / cell)), int(math.floor(hi[i][1] / cell)) + 1):
                grid[(gx, gz)].append(i)
    rows = []
    for p in parts:
        pts = np.asarray(pts_by_part.get(p['id']) or [], float).reshape(-1, 4)    # x, y, z, block id
        if not len(pts):
            rows.append(dict(id=p['id'], kind=p['kind'], n=0, blocks=0, verdict='incomplete', reason='no withheld points'))
            continue
        res, missing, multiple = [], 0, 0
        for x, y, z, _b in pts:
            hits = []
            for i in grid.get((int(math.floor(x / cell)), int(math.floor(z / cell))), ()):
                (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri[i]
                d = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
                if abs(d) < 1e-12: continue
                l1 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / d
                l2 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / d
                l3 = 1 - l1 - l2
                if min(l1, l2, l3) >= -1e-9: hits.append(l1 * ay + l2 * by + l3 * cy)
            hs = sorted(set(round(v, 3) for v in hits))
            if not hs: missing += 1; continue
            if len(hs) > 1 and hs[-1] - hs[0] > 0.02: multiple += 1
            res.append(y - max(hs))                       # + : point above the roof
        res = np.asarray(res)
        blocks = int(len(np.unique(pts[:, 3])))
        row = dict(id=p['id'], kind=p['kind'], n=int(len(pts)), measured=int(len(res)), missing_hits=missing, multiple_hits=multiple,
                   blocks=blocks, block_area_ft2=blocks * 100,
                   note='block_area_ft2 is the area of withheld 10 ft blocks, not measured roof coverage')
        if len(res):
            ab = np.abs(res)
            row.update(median_abs=round(float(np.median(ab)), 2), p90_abs=round(float(np.percentile(ab, 90)), 2),
                       signed_p10=round(float(np.percentile(res, 10)), 2), signed_p90=round(float(np.percentile(res, 90)), 2))
        if len(res) < MIN_HELDOUT or blocks < MIN_BLOCKS: row['verdict'] = 'incomplete'
        elif multiple: row['verdict'] = 'fail'; row['reason'] = f'{multiple} withheld points see more than one roof surface (stacked or duplicate roofs)'
        elif missing > 0.05 * len(pts): row['verdict'] = 'fail'; row['reason'] = 'withheld points with no roof above or below them'
        elif row['median_abs'] <= GATE_MEDIAN and row['p90_abs'] <= GATE_P90: row['verdict'] = 'pass'
        else: row['verdict'] = 'fail'
        rows.append(row)
    return rows


def run(folder):
    S = json.load(open(os.path.join(folder, 'scene.json')))
    terr = terrain_from(folder)
    hp = os.path.join(folder, 'heldout.json')
    heldout = json.load(open(hp)) if os.path.exists(hp) else None
    houses = S.get('houses', [])
    checked = [check_building(h, terr, heldout) for h in houses if h.get('mesh')]
    not_checked = [h['id'] for h in houses if not h.get('mesh')]
    acc = verdict(*[b['acceptance'] for b in checked]) if checked else 'not_applicable'
    if not_checked: acc = verdict(acc, 'incomplete')
    res = dict(thresholds=dict(heldout_median_ft=GATE_MEDIAN, heldout_p90_ft=GATE_P90, min_heldout_points=MIN_HELDOUT,
                               min_heldout_blocks=MIN_BLOCKS, base_tol_ft=BASE_TOL,
                               note='provisional development gates (C4); not validated accuracy'),
               population=dict(buildings=len(houses), checked=len(checked), not_checked=not_checked),
               topology_only=verdict(*[b['topology_only'] for b in checked]) if checked else 'not_applicable',
               acceptance=acc, buildings=checked)
    json.dump(res, open(os.path.join(folder, 'validation.json'), 'w'), indent=1)
    for b in checked:
        t = b['topology']; hv = [r['verdict'] for r in b['heldout']['parts']]
        print(f"{b['id']:9s} {str(b['name'] or '')[:24]:24s} {b['acceptance']:10s}| topo {t['verdict']:4s} base {b['base']['verdict']:10s} "
              f"orient {b['orientation']['verdict']:4s}({b['orientation']['inverted']} inv) inter {b['intersections']['verdict']:4s}"
              f"({b['intersections']['crossings']}x,{b['intersections']['duplicates']}d) | held-out {b['heldout']['verdict']:10s} "
              f"pass {hv.count('pass')} fail {hv.count('fail')} incomplete {hv.count('incomplete')}")
    print(f"site acceptance: {acc}  (checked {len(checked)} of {len(houses)}; not checked: {not_checked or 'none'}); topology-only: {res['topology_only']}")
    return res


if __name__ == '__main__':
    r = run(sys.argv[1])
    sys.exit(0 if r['acceptance'] == 'pass' else 1)
