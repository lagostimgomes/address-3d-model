#!/usr/bin/env python3
"""
build_model.py - write a self-contained Three.js page (index.html) for a site folder that
pipeline/fetch/fetch_site.py and pipeline/build/analyze_site.py have filled.

    python build_model.py data/<slug> [--out data/<slug>/index.html]

Reads site.json, dem.json, scene.json and, when present, roads.json and osm_buildings.json.
Standard library only.
"""
import argparse, json, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PALETTE = ['#cbc5b6', '#a6a8a3', '#c4c5bf', '#d9d4c7', '#b9b2a2', '#9fa7ad', '#e2ddd2', '#b7ae9c']
DOORS = ['#2a2d31', '#1d1f22', '#5b2a26', '#24384a', '#f2f1ec']
GARAGE = ['#f0efea', '#e8e6df', '#5d7d97', '#2f4636', '#1e2023']


def h32(s):
    v = 2166136261
    for ch in s.encode():
        v = ((v ^ ch) * 16777619) & 0xffffffff
    return v


def house_number(addr):
    m = re.match(r'\s*(\d+[A-Za-z]?)\b', addr or '')
    return m.group(1) if m else None


def label_neighbors(scene, folder):
    p = os.path.join(folder, 'osm_buildings.json')
    if not os.path.exists(p): return
    data = json.load(open(p))
    blds = data.get('buildings', []) if isinstance(data, dict) else data
    for hh in scene['houses']:
        if hh['role'] != 'neighbor': continue
        best = None
        for b in blds:
            num = (b.get('tags') or {}).get('addr:housenumber') or b.get('housenumber')
            pts = b.get('points') or b.get('pts')
            if not num or not pts: continue
            cx = sum(p[0] for p in pts) / len(pts); cz = sum(p[1] for p in pts) / len(pts)
            d = math.hypot(cx - hh['x'], cz - hh['z'])
            if d < 30 and (best is None or d < best[0]): best = (d, num, (b.get('tags') or {}).get('addr:street'))
        if best: hh['number'] = best[1]; hh['street'] = best[2]


def build(folder, out=None):
    site = json.load(open(os.path.join(folder, 'site.json')))
    dem = json.load(open(os.path.join(folder, 'dem.json')))
    scene = json.load(open(os.path.join(folder, 'scene.json')))
    roads = []
    rp = os.path.join(folder, 'roads.json')
    if os.path.exists(rp):
        r = json.load(open(rp)); r = r.get('roads', []) if isinstance(r, dict) else r
        for w in r:
            pts = w.get('points') or w.get('pts') or []
            hw = w.get('highway') or (w.get('tags') or {}).get('highway') or 'residential'
            if len(pts) >= 2 and hw not in ('footway', 'path', 'steps', 'cycleway', 'bridleway', 'pedestrian', 'track', 'corridor', 'proposed', 'construction'):
                roads.append(dict(hw=hw, name=w.get('name') or (w.get('tags') or {}).get('name'), pts=[[round(p[0], 1), round(p[1], 1)] for p in pts]))
    label_neighbors(scene, folder)
    lp = os.path.join(folder, 'locate.json'); loc = json.load(open(lp)) if os.path.exists(lp) else {}
    pad = scene['pad_ft']
    grid = ','.join(str(int(round((v - pad) * 10))) for v in dem['elevations_ft'])
    num = house_number(site['address']) or 'Home'
    street_name = scene['street'].get('name')
    houses = []
    for hh in scene['houses']:
        key = f"{site['address']}|{hh['id']}"; hv = h32(key)
        lbl = num if hh['role'] == 'featured' else (hh.get('number') or ('West' if False else ('Neighbor')))
        houses.append(dict(hh, label=lbl, siding=PALETTE[hv % len(PALETTE)], door=DOORS[(hv >> 4) % len(DOORS)],
                           garage=GARAGE[(hv >> 8) % len(GARAGE)], garage_side=1 if (hv >> 12) % 3 else -1))
    feat = next(h for h in houses if h['role'] == 'featured')
    m = re.match(r'\s*([^,]+),\s*(.*)$', loc.get('matched_address') or site.get('matched_address') or site['address'])
    title = m.group(1).strip() if m else site['address']
    sub = m.group(2).strip() if m else ''
    years = sorted({d['year'] for d in scene['datasets'] if d['year']}, reverse=True)
    data = dict(title=title, sub=sub, lat=site['lat'], lon=site['lon'], pad=pad, n=dem['rows'], cell=dem['cell_size_ft'],
                houses=houses, trees=scene['trees'], roads=roads, front=scene['street']['front'], pavement=scene.get('pavement'),
                street_source=scene['street']['source'], street=street_name,
                lidar=', '.join(f"{d['name']} ({d['year'] or 'n.d.'})" for d in scene['datasets']),
                lidar_years=' and '.join(str(y) for y in years), precision=loc.get('precision', 'given'),
                loc_source=loc.get('source', 'coordinates'), dem_src=(dem.get('source') or {}).get('service_url') or (dem.get('source') or {}).get('base_url') or 'USGS 3DEP')
    tpl = open(os.path.join(HERE, 'template.html'), encoding='utf-8').read()
    html = tpl.replace('__TITLE__', title).replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':'))).replace('__GRID__', grid)
    out = out or os.path.join(folder, 'index.html')
    open(out, 'w', encoding='utf-8').write(html)
    print(f'wrote {out} ({len(html)/1024:.0f} KB): {len(houses)} houses, {len(scene["trees"])} trees, {len(roads)} roads')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder'); ap.add_argument('--out')
    a = ap.parse_args(); build(a.folder, a.out)
