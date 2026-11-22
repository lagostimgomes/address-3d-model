#!/usr/bin/env python3
"""
identify.py - name the buildings in a site model, with no human involved.

    python identify.py sites/<slug>                       # institutions / sites around the building
    python identify.py sites/<slug> --names "A" "B" ...   # verify candidate building names

1. Institutions: Nominatim structured search for amenities (university, school, hospital, place of
   worship, library, ...) in a box around the featured building -> name, website, Wikidata id.
2. Verification: each candidate name (found by web search, campus maps, etc.) is looked up in
   OpenStreetMap inside the box. It is accepted only if the OSM feature lies on one of the measured
   buildings (inside its footprint, or within 20 ft of it). OSM tags such as building:levels,
   building:material and roof:shape are kept.
Writes identify.json; prints it. Standard library only. Uses nominatim.openstreetmap.org (1 req/s).
"""
import argparse, json, math, os, time, urllib.parse, urllib.request

UA = {'User-Agent': 'address-3d-model/1.0 (https://github.com/lagostimgomes/address-3d-model)'}
AMENITIES = ['university', 'college', 'school', 'hospital', 'place_of_worship', 'library', 'townhall',
             'courthouse', 'community_centre', 'arts_centre', 'theatre', 'fire_station']
KEEP = ('building', 'roof', 'height', 'min_height', 'start_date', 'architect', 'wikidata', 'wikipedia', 'website', 'heritage')
R = 6378137.0


def get(url, params):
    time.sleep(1.1)
    with urllib.request.urlopen(urllib.request.Request(url + '?' + urllib.parse.urlencode(params), headers=UA), timeout=30) as r:
        return json.loads(r.read())


class Frame:
    def __init__(self, site):
        self.cx, self.cy = site['mercator_center']; self.k = math.cos(math.radians(site['lat'])) * 3.28084

    def to_ll(self, x, z):
        X = self.cx + x / self.k; Y = self.cy - z / self.k
        return math.degrees(2 * math.atan(math.exp(Y / R)) - math.pi / 2), math.degrees(X / R)

    def to_xz(self, lat, lon):
        X = math.radians(lon) * R; Y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        return (X - self.cx) * self.k, -(Y - self.cy) * self.k


def on_building(x, z, b, slack=20.0):
    """Is (x, z) on building b? Grid buildings: inside the labelled grid; houses: inside the rotated box."""
    if b['roof']['type'] == 'grid':
        G = b['grid']; ca, sa = math.cos(G['ang']), math.sin(G['ang'])
        u, v = x * ca + z * sa, -x * sa + z * ca
        j = (u - G['x0']) / G['cell']; i = (v - G['z0']) / G['cell']
        pad = slack / G['cell']
        for r in range(max(0, int(i - pad)), min(len(G['rows']), int(i + pad) + 1)):
            runs = G['rows'][r]
            for k in range(0, len(runs), 3):
                if runs[k] - pad <= j <= runs[k] + runs[k + 1] + pad: return True
        return False
    r = b['rot']; dx, dz = x - b['x'], z - b['z']
    lx = dx * math.cos(r) - dz * math.sin(r); lz = dx * math.sin(r) + dz * math.cos(r)
    return abs(lx) <= b['w'] / 2 + slack and abs(lz) <= b['d'] / 2 + slack


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder'); ap.add_argument('--names', nargs='*', default=[])
    ap.add_argument('--radius', type=float, default=350, help='box half-size for verifying building names, ft')
    ap.add_argument('--site-radius', type=float, default=1500, help='box half-size for institutions (campuses are large), ft')
    a = ap.parse_args()
    site = json.load(open(os.path.join(a.folder, 'site.json'))); scene = json.load(open(os.path.join(a.folder, 'scene.json')))
    F = Frame(site); feat = next(b for b in scene['houses'] if b['role'] == 'featured')
    la1, lo1 = F.to_ll(feat['x'] - a.radius, feat['z'] - a.radius); la2, lo2 = F.to_ll(feat['x'] + a.radius, feat['z'] + a.radius)
    vb = f'{min(lo1, lo2):.6f},{max(la1, la2):.6f},{max(lo1, lo2):.6f},{min(la1, la2):.6f}'   # left,top,right,bottom
    la1, lo1 = F.to_ll(feat['x'] - a.site_radius, feat['z'] - a.site_radius); la2, lo2 = F.to_ll(feat['x'] + a.site_radius, feat['z'] + a.site_radius)
    vb_site = f'{min(lo1, lo2):.6f},{max(la1, la2):.6f},{max(lo1, lo2):.6f},{min(la1, la2):.6f}'
    out = dict(viewbox=vb, institutions=[], buildings={}, rejected=[])
    path = os.path.join(a.folder, 'identify.json')
    if os.path.exists(path):
        old = json.load(open(path)); out['institutions'] = old.get('institutions', []); out['buildings'] = old.get('buildings', {})
    if not out['institutions']:
        for am in AMENITIES:
            try: res = get('https://nominatim.openstreetmap.org/search', dict(format='jsonv2', amenity=am, viewbox=vb_site, bounded=1, limit=5, extratags=1))
            except Exception as e: res = []; out.setdefault('errors', []).append(f'{am}: {e}')
            for d in res:
                if not d.get('name'): continue
                x, z = F.to_xz(float(d['lat']), float(d['lon']))
                t = d.get('extratags') or {}
                out['institutions'].append(dict(name=d['name'], amenity=am, osm=f"{d.get('osm_type')}/{d.get('osm_id')}",
                                                distance_ft=round(math.hypot(x - feat['x'], z - feat['z'])),
                                                tags={k: v for k, v in t.items() if k.startswith(KEEP)}))
        out['institutions'].sort(key=lambda i: i['distance_ft'])
        out['institutions'] = out['institutions'][:8]
    for name in a.names:
        try: res = get('https://nominatim.openstreetmap.org/search', dict(format='jsonv2', q=name, viewbox=vb, bounded=1, limit=5, extratags=1))
        except Exception as e: out['rejected'].append(dict(name=name, reason=str(e))); continue
        hit = None
        for d in res:
            x, z = F.to_xz(float(d['lat']), float(d['lon']))
            for b in scene['houses']:
                if on_building(x, z, b):
                    t = d.get('extratags') or {}
                    hit = (b['id'], dict(name=d.get('name') or name, osm=f"{d.get('osm_type')}/{d.get('osm_id')}", osm_class=f"{d.get('category')}={d.get('type')}",
                                         tags={k: v for k, v in t.items() if k.startswith(KEEP)}))
                    break
            if hit: break
        if hit and hit[0] not in out['buildings']: out['buildings'][hit[0]] = hit[1]
        elif not hit: out['rejected'].append(dict(name=name, reason='not on a modelled building' if res else 'not found in OSM near the site'))
    json.dump(out, open(path, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
