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
import sys; sys.path.insert(0, HERE)
from analyze_site import dem_bounds  # noqa: E402
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


def ground_features(folder, scene):
    """Mapped ground for the page: road/path centrelines with widths and surface polygons, clipped to the
    DEM rectangle. Widths come from the map (width tag, else lanes), else a default for the road class,
    and each way says which (the page lists defaults as generated)."""
    from analyze_site import load_features
    from shapely.geometry import LineString, Polygon, box
    F = load_features(folder)
    b = scene.get('dem_bounds_ft'); bb = box(b[0], b[2], b[1], b[3])
    ways, surfaces = [], []
    for w in F.get('ways', []):
        k = w['kind']
        if k in ('proposed', 'construction', 'corridor', 'elevator', 'platform', 'bus_stop', 'raceway') or w.get('tunnel') in ('yes', 'building_passage'): continue
        try: ls = LineString(w['centerline']).intersection(bb)
        except Exception: continue
        if ls.is_empty: continue
        wf, src = w.get('width_ft'), w.get('width_source')
        if not wf and w.get('lanes'):
            try: wf = float(w['lanes']) * 11 + (2 if k in VEH_ROADS else 0); src = 'lanes'
            except ValueError: pass
        if not wf:
            wf = default_width(k, w); src = 'default'
        for g in getattr(ls, 'geoms', [ls]):
            if g.geom_type == 'LineString' and g.length > 1:
                ways.append(dict(k=k, w=round(wf, 1), ws=src, f=w.get('footway'), sv=w.get('service'), br=1 if w.get('bridge') == 'yes' else 0,
                                 pts=[[round(x, 1), round(z, 1)] for x, z in g.coords]))
    for sf in F.get('surfaces', []):
        try: pg = Polygon(sf['polygon'], sf.get('holes') or []).buffer(0).intersection(bb)
        except Exception: continue
        for g in getattr(pg, 'geoms', [pg]):
            if g.geom_type == 'Polygon' and g.area > 20:
                surfaces.append(dict(k=sf['kind'], ring=[[round(x, 1), round(z, 1)] for x, z in g.exterior.coords[:-1]],
                                     holes=[[[round(x, 1), round(z, 1)] for x, z in h.coords[:-1]] for h in g.interiors]))
    order = {'lawn': 0, 'pitch': 1, 'parking': 2, 'plaza': 3, 'road_area': 4, 'other_paved': 4, 'pool': 5, 'water': 5}
    surfaces.sort(key=lambda s_: order.get(s_['k'], 3))
    return dict(ways=ways, surfaces=surfaces)


VEH_ROADS = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street'}
TOTAL_W = {'motorway': 48, 'trunk': 44, 'primary': 40, 'secondary': 34, 'tertiary': 30, 'residential': 28, 'unclassified': 22,
           'living_street': 18, 'motorway_link': 18, 'trunk_link': 18, 'primary_link': 18, 'secondary_link': 18, 'tertiary_link': 18,
           'pedestrian': 14, 'track': 10, 'footway': 5, 'path': 5, 'cycleway': 6, 'bridleway': 6, 'steps': 6}


def default_width(k, w):
    if k == 'service':
        return {'driveway': 10, 'parking_aisle': 22, 'alley': 12, 'drive-through': 11}.get(w.get('service'), 14)
    if k == 'footway' and w.get('footway') == 'crossing': return 8
    return TOTAL_W.get(k, 12)


def build(folder, out=None):
    site = json.load(open(os.path.join(folder, 'site.json')))
    dem = json.load(open(os.path.join(folder, 'dem.json')))
    scene = json.load(open(os.path.join(folder, 'scene.json')))
    ground = ground_features(folder, scene)
    if not os.path.exists(os.path.join(folder, 'features.json')): label_neighbors(scene, folder)
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
    data = dict(title=title, sub=sub, lat=site['lat'], lon=site['lon'], pad=pad, rows=dem['rows'], cols=dem.get('cols', dem['rows']),
                cell=dem['cell_size_ft'], bounds=scene.get('dem_bounds_ft') or dem_bounds(dem), coverage=scene.get('coverage'),
                houses=houses, trees=scene['trees'], ground=ground, front=scene['street']['front'], pavement=scene.get('pavement'),
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
    print(f'wrote {out} ({len(html)/1024:.0f} KB): {len(houses)} buildings, {len(scene["trees"])} trees, {len(ground["ways"])} ways, {len(ground["surfaces"])} surfaces')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder'); ap.add_argument('--out')
    a = ap.parse_args(); build(a.folder, a.out)
