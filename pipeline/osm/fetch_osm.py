#!/usr/bin/env python3
"""
fetch_osm.py - streets and building footprints from OpenStreetMap (Overpass API) for a site
folder written by pipeline/fetch/fetch_site.py.

    python fetch_osm.py data/<slug> [--roads 220] [--buildings 140]

Writes roads.json and osm_buildings.json in the model frame (feet, +x east, +z south, origin at
site.json mercator_center). Never fails the pipeline: on error it writes {"status": "unavailable"}.
Standard library only. Data (c) OpenStreetMap contributors, ODbL.
"""
import argparse, json, math, os, sys, time, urllib.parse, urllib.request

ENDPOINTS = ['https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter',
             'https://overpass.private.coffee/api/interpreter']
R = 6378137.0
FT = 3.28084


def to_model(lat, lon, site):
    x = math.radians(lon) * R
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    cx, cy = site['mercator_center']; k = math.cos(math.radians(site['lat'])) * FT
    return [round((x - cx) * k, 2), round(-(y - cy) * k, 2)]


def overpass(query, log):
    last = None
    for url in ENDPOINTS:
        for attempt in range(2):
            try:
                data = urllib.parse.urlencode({'data': query}).encode()
                req = urllib.request.Request(url, data=data, headers={'User-Agent': 'address-3d-model/1.0'})
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read()
                log.append(f'POST {url} -> {len(raw)} bytes')
                return json.loads(raw), url
            except Exception as e:
                last = e; log.append(f'POST {url} attempt {attempt + 1} failed: {type(e).__name__}: {e}')
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'all Overpass endpoints failed: {last}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder'); ap.add_argument('--roads', type=float, default=220); ap.add_argument('--buildings', type=float, default=140)
    a = ap.parse_args()
    site = json.load(open(os.path.join(a.folder, 'site.json')))
    lat, lon = site['lat'], site['lon']
    q = (f'[out:json][timeout:50];('
         f'way["highway"](around:{a.roads:.0f},{lat},{lon});'
         f'way["building"](around:{a.buildings:.0f},{lat},{lon});'
         f');out tags geom;')
    log = [f'query: {q}']
    try:
        data, url = overpass(q, log)
    except Exception as e:
        for name in ('roads.json', 'osm_buildings.json'):
            json.dump({'status': 'unavailable', 'error': str(e), 'log': log}, open(os.path.join(a.folder, name), 'w'), indent=1)
        print(f'OSM unavailable: {e}'); return 0
    roads, blds = [], []
    for el in data.get('elements', []):
        if el.get('type') != 'way' or 'geometry' not in el: continue
        t = el.get('tags', {}); pts = [to_model(g['lat'], g['lon'], site) for g in el['geometry']]
        if 'highway' in t:
            roads.append({'id': el['id'], 'highway': t['highway'], 'name': t.get('name'), 'points': pts})
        elif 'building' in t:
            blds.append({'id': el['id'], 'tags': {k: v for k, v in t.items() if k.startswith(('addr:', 'building', 'height', 'roof'))}, 'points': pts})
    meta = {'status': 'ok', 'source': 'OpenStreetMap via Overpass', 'endpoint': url, 'query': q,
            'license': 'ODbL, (c) OpenStreetMap contributors', 'log': log}
    json.dump({**meta, 'roads': roads}, open(os.path.join(a.folder, 'roads.json'), 'w'), indent=1)
    json.dump({**meta, 'buildings': blds}, open(os.path.join(a.folder, 'osm_buildings.json'), 'w'), indent=1)
    print(f'OSM: {len(roads)} ways with highway tags, {len(blds)} buildings ({url})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
