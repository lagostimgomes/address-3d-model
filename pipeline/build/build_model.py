#!/usr/bin/env python3
"""
build_model.py - write a self-contained Three.js page (index.html) for a site folder that
pipeline/fetch/fetch_site.py and pipeline/build/analyze_site.py have filled.

    python build_model.py sites/<slug> [--out sites/<slug>/index.html]

Reads site.json, dem.json, scene.json and, when present, roads.json, osm_buildings.json,
identify.json (names and OSM tags, from pipeline/enrich/identify.py) and facts.json.

facts.json (optional, written by the enrichment step; every fact needs a source):
{
  "site":      {"name": "...", "wall": {"material": "brick"}, "roof": {"material": "slate"},
                "style": "georgian", "sources": ["https://..."]},          # defaults for all buildings
  "buildings": {"featured": {"name": "...", "levels": 2, "wall": {"material": "brick", "color": "#8f4e3b"},
                             "roof": {"material": "slate"}, "style": "georgian", "basis": "building",
                             "summary": "one line", "sources": ["https://..."]},
                "b0": {...}}                                                # ids from scene.json
}
wall/roof materials map to colours below; an explicit "color" wins. style: georgian | colonial |
traditional (punched windows with white trim) or modern | glass (ribbon windows). basis: "building"
(facts about this building) or "site" (campus- or district-level facts applied to it).
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


WALL = {'red brick': '#8f4e3b', 'brick': '#8f4e3b', 'brown brick': '#6e4a3a', 'tan brick': '#c2a27a', 'buff brick': '#c9b28a',
        'yellow brick': '#cdb37a', 'white brick': '#e4e0d6', 'stone': '#bdb5a6', 'fieldstone': '#a69a87', 'limestone': '#d8d0bd',
        'sandstone': '#c8ab82', 'brownstone': '#7b5a46', 'granite': '#a9a7a3', 'marble': '#e6e3dc', 'concrete': '#b7b5af',
        'precast': '#c7c3b9', 'stucco': '#e2d9c6', 'wood': '#a8865f', 'siding': '#cbc5b6', 'vinyl': '#d6d2c6',
        'glass': '#7f97a8', 'metal': '#a3a7ab', 'log': '#8a6a45', 'adobe': '#c79a6e'}
ROOF = {'slate': '#4b5058', 'asphalt': '#55585c', 'shingle': '#55585c', 'copper': '#5e9e88', 'metal': '#8a9096',
        'standing seam': '#7d838a', 'tile': '#a45a3c', 'clay': '#a45a3c', 'membrane': '#9a9c9e', 'epdm': '#3c3d3f',
        'tpo': '#d9dbdc', 'gravel': '#8c8a84', 'green': '#5a7a3a', 'wood': '#7a6248', 'tar': '#3f4144'}
MODERN = ('modern', 'glass', 'contemporary', 'international', 'brutalist')


def pick(table, mat):
    if not mat: return None
    m = mat.lower()
    for k in sorted(table, key=len, reverse=True):
        if k in m: return table[k]
    return None


def apply_facts(houses, folder):
    """Names and levels from identify.json; materials, colours and style from facts.json."""
    ident = json.load(open(os.path.join(folder, 'identify.json'))) if os.path.exists(os.path.join(folder, 'identify.json')) else {}
    facts = json.load(open(os.path.join(folder, 'facts.json'))) if os.path.exists(os.path.join(folder, 'facts.json')) else {}
    site = facts.get('site') or {}
    used = []
    for h in houses:
        idn = (ident.get('buildings') or {}).get(h['id']) or {}
        f = (facts.get('buildings') or {}).get(h['id']) or {}
        tags = idn.get('tags') or {}
        name = f.get('name') or idn.get('name')
        if name: h['name'] = name
        lv = f.get('levels') or tags.get('building:levels')
        try: lv = int(float(lv)) if lv else None
        except ValueError: lv = None
        if lv and 1 <= lv <= 80:
            h['levels'] = lv; h['stories'] = lv
            h['floor_h'] = round(max(8.5, min(18.0, h['wall_h'] / lv)), 2)
        wall = (f.get('wall') or {}) or (site.get('wall') or {})
        roof = (f.get('roof') or {}) or (site.get('roof') or {})
        wc = wall.get('color') or pick(WALL, wall.get('material')) or pick(WALL, tags.get('building:material'))
        rc = roof.get('color') or pick(ROOF, roof.get('material')) or pick(ROOF, tags.get('roof:material'))
        if tags.get('building:colour', '').startswith('#'): wc = wc or tags['building:colour']
        if tags.get('roof:colour', '').startswith('#'): rc = rc or tags['roof:colour']
        if wc: h['wall_color'] = wc; h['siding'] = wc
        if rc: h['roof_color'] = rc
        style = (f.get('style') or site.get('style') or '').lower()
        if style: h['facade_style'] = 'ribbon' if any(k in style for k in MODERN) else 'punched'
        basis = f.get('basis') or ('site' if (wall or roof or style) and not f else 'building')
        srcs = f.get('sources') or (site.get('sources') if not f.get('wall') else []) or []
        if f.get('summary') or wc or rc or lv:
            h['facts'] = dict(summary=f.get('summary'), basis=basis if (wall or roof or style) else None,
                              wall=wall.get('material'), roof=roof.get('material'), style=f.get('style') or site.get('style'),
                              levels=lv, sources=srcs)
        used += srcs
    return dict(site_name=site.get('name') or next((i['name'] for i in ident.get('institutions', [])
                                                    if i.get('amenity') in ('university', 'college', 'school', 'hospital')), None),
                sources=sorted(set(used + (site.get('sources') or []))))


def house_number(addr):
    m = re.match(r'\s*(\d+[A-Za-z]?)\b', addr or '')
    return m.group(1) if m else None


def label_neighbors(scene, folder):
    p = os.path.join(folder, 'osm_buildings.json')
    if not os.path.exists(p): return
    data = json.load(open(p))
    blds = data.get('buildings', []) if isinstance(data, dict) else data
    for hh in scene['houses']:
        if hh['role'] == 'featured': continue
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
        lbl = num if hh['role'] == 'featured' else (hh.get('number') or '')
        houses.append(dict(hh, label=lbl, hash=hv, siding=PALETTE[hv % len(PALETTE)], door=DOORS[(hv >> 4) % len(DOORS)],
                           garage=GARAGE[(hv >> 8) % len(GARAGE)], garage_side=1 if (hv >> 12) % 3 else -1))
    enr = apply_facts(houses, folder)
    feat = next(h for h in houses if h['role'] == 'featured')
    m = re.match(r'\s*([^,]+),\s*(.*)$', loc.get('matched_address') or site.get('matched_address') or site['address'])
    title = m.group(1).strip() if m else site['address']
    sub = m.group(2).strip() if m else ''
    years = sorted({d['year'] for d in scene['datasets'] if d['year']}, reverse=True)
    data = dict(title=title, sub=sub, lat=site['lat'], lon=site['lon'], pad=pad, n=dem['rows'], cell=dem['cell_size_ft'],
                houses=houses, trees=scene['trees'], roads=roads, front=scene['street']['front'], pavement=scene.get('pavement'),
                site=scene['site'], outline=scene.get('outline'), buffer_ft=scene.get('buffer_ft', 100),
                site_name=enr['site_name'], fact_sources=enr['sources'], feat_name=feat.get('name'),
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
