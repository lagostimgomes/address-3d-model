"""
gates.py - machine checks on the EXPORTED (quantized) building meshes (contract C4).

For each part-built building in scene.json: decode the int32 0.01 ft positions exactly as the page does, weld at
the encoding precision (identical integer coordinates only; distinct surfaces are never merged), then count:
- degenerate triangles (zero area after quantization)
- T-junctions (a welded vertex lying strictly inside another triangle's edge)
- boundary edges (used by one triangle) that are NOT on the building's allowed open loop: the terrain base,
  i.e. both ends at the bottom of a ground wall
- non-manifold edges (used by more than two triangles)
- orientation conflicts (an interior edge traversed in the same direction by both of its triangles)
plus the held-out roof residuals per part (computed in the build before segmentation), with underpowered parts flagged.
Thresholds on residuals are provisional development gates (C4), not validated accuracy.

Usage: python gates.py <site folder>   -> writes <folder>/validation.json, prints a summary, exit 1 on a failed gate.
"""
import base64, json, os, sys
from collections import defaultdict
import numpy as np

GATE_MEDIAN, GATE_P90, MIN_HELDOUT = 0.8, 2.5, 50


def decode(m):
    if not m or not m.get('n'): return np.zeros((0, 3), np.int64)
    a = np.frombuffer(base64.b64decode(m['p']), '<i4').astype(np.int64)
    return a.reshape(-1, 3)


def check_building(h):
    M = h['mesh']
    P = np.concatenate([decode(M.get(k)) for k in ('roof', 'roof_flat', 'walls', 'section')])
    tris = P.reshape(-1, 3, 3)
    # weld: identical quantized coordinates
    keys, inv = np.unique(P, axis=0, return_inverse=True)
    T = inv.reshape(-1, 3)
    V = keys
    # degenerate
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cr = np.cross((b - a).astype(float), (c - a).astype(float))
    area2 = np.linalg.norm(cr, axis=1)
    degenerate = int((area2 < 1.0).sum())            # < 0.5 (0.01 ft)^2 units: zero after quantization
    good = area2 >= 1.0
    T = T[good]
    edges = defaultdict(list)
    for t in T:
        for i in range(3):
            u, v = int(t[i]), int(t[(i + 1) % 3])
            edges[(min(u, v), max(u, v))].append((u, v))
    # the allowed open loop is declared by the build (C4): bottom edges of ground walls (terrain base)
    kid = {tuple(k): i for i, k in enumerate(V.tolist())}
    allowed = set()
    for a_, b_ in M.get('base') or []:
        qa = tuple(int(round(c * 100)) for c in a_); qb = tuple(int(round(c * 100)) for c in b_)
        if qa in kid and qb in kid: allowed.add((min(kid[qa], kid[qb]), max(kid[qa], kid[qb])))
    boundary = nonmanifold = orient = base_edges = 0
    for (u, v), uses in edges.items():
        if len(uses) == 1:
            if (u, v) in allowed: base_edges += 1
            else: boundary += 1
        elif len(uses) > 2: nonmanifold += 1
        elif uses[0] == uses[1]: orient += 1
    # T-junctions: a vertex strictly inside a boundary edge (only boundary edges can hide them after welding)
    tj = 0
    if boundary:
        bd = [(u, v) for (u, v), us in edges.items() if len(us) == 1 and (u, v) not in allowed]
        Vf = V.astype(float)
        for u, v in bd[:5000]:
            p, q = Vf[u], Vf[v]; d = q - p; L2 = d @ d
            if L2 < 1: continue
            t = ((Vf - p) @ d) / L2
            near = (t > 1e-4) & (t < 1 - 1e-4)
            if near.any():
                proj = p + np.outer(t[near], d)
                if (np.linalg.norm(Vf[near] - proj, axis=1) < 1.0).any(): tj += 1
    parts = h.get('parts') or []
    ho = []
    for p in parts:
        H = p.get('heldout') or {}
        row = dict(id=p['id'], kind=p['kind'], n=H.get('n'), median_abs=H.get('median_abs'), p90_abs=H.get('p90_abs'))
        if not H.get('n') or H['n'] < MIN_HELDOUT: row['verdict'] = 'underpowered'
        elif H['median_abs'] <= GATE_MEDIAN and H['p90_abs'] <= GATE_P90: row['verdict'] = 'pass'
        else: row['verdict'] = 'fail'
        ho.append(row)
    topo_ok = degenerate == 0 and boundary == 0 and nonmanifold == 0 and orient == 0
    return dict(id=h['id'], name=h.get('name'), triangles=int(len(T)), vertices=int(len(V)),
                topology=dict(degenerate=degenerate, open_edges_not_base=boundary, base_edges=base_edges,
                              nonmanifold=nonmanifold, orientation_conflicts=orient, t_junctions=tj, ok=topo_ok),
                heldout=ho, heldout_ok=all(r['verdict'] != 'fail' for r in ho),
                envelopes=sum(1 for p in parts if p['kind'] == 'envelope'), parts=len(parts))


def run(folder):
    S = json.load(open(os.path.join(folder, 'scene.json')))
    out = [check_building(h) for h in S['houses'] if h.get('type') == 'mesh' and h.get('mesh', {}).get('roof', {}).get('q')]
    res = dict(thresholds=dict(heldout_median_ft=GATE_MEDIAN, heldout_p90_ft=GATE_P90, min_heldout_points=MIN_HELDOUT,
                               note='provisional development gates (C4); not validated accuracy'),
               buildings=out, ok=all(b['topology']['ok'] for b in out))
    json.dump(res, open(os.path.join(folder, 'validation.json'), 'w'), indent=1)
    for b in out:
        t = b['topology']; hv = [r['verdict'] for r in b['heldout']]
        print(f"{b['id']:9s} {str(b['name'] or '')[:26]:26s} tris {b['triangles']:6d} | open {t['open_edges_not_base']:4d} nonmanifold {t['nonmanifold']:3d} "
              f"orient {t['orientation_conflicts']:3d} degenerate {t['degenerate']:3d} Tj {t['t_junctions']:3d} | parts {b['parts']:2d} "
              f"envelopes {b['envelopes']:2d} heldout pass {hv.count('pass')} fail {hv.count('fail')} underpowered {hv.count('underpowered')}")
    return res


if __name__ == '__main__':
    r = run(sys.argv[1])
    sys.exit(0 if r['ok'] else 1)
