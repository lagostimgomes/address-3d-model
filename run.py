#!/usr/bin/env python3
"""
run.py - address in, 3D neighborhood model out.

    python run.py "1600 Pennsylvania Ave NW, Washington, DC 20500" [--lat LAT --lon LON] [--work DIR]

Steps: network preflight -> pipeline/fetch/fetch_site.py (geocode, terrain, LiDAR) -> pipeline/osm/fetch_osm.py (streets)
-> pipeline/build/analyze_site.py (houses, roofs, trees) -> pipeline/build/build_model.py (index.html).
Prints one JSON line at the end starting with RESULT: {...}.
Exit codes: 0 ok, 2 bad input, 3 network blocked, 4 fetch failed, 5 analysis failed, 6 build failed.
"""
import argparse, json, os, subprocess, sys, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
# (host, test URL, required?)
HOSTS = [
    ('geocoding.geo.census.gov', 'https://geocoding.geo.census.gov/geocoder/benchmarks', False),
    ('nominatim.openstreetmap.org', 'https://nominatim.openstreetmap.org/status', False),
    ('raw.githubusercontent.com', 'https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/README.md', True),
    ('s3-us-west-2.amazonaws.com', 'https://s3-us-west-2.amazonaws.com/usgs-lidar-public/MD_Western_2_D21/ept.json', True),
    ('elevation.nationalmap.gov', 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer?f=json', True),
    ('overpass-api.de', 'https://overpass-api.de/api/status', False),
]


def probe(url):
    """True if the host answers at all (any HTTP status, even 5xx: that is the service, not a block).
    False only when the connection itself is refused, e.g. by a sandbox egress proxy."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'address-3d-model/1.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            return True
    except urllib.error.HTTPError as e:
        return e.code != 407
    except Exception:
        return False


def preflight(skip_geocode):
    res = {h: probe(u) for h, u, _ in HOSTS}
    blocked_req = [h for h, _, req in HOSTS if req and not res[h]]
    geo_ok = res['geocoding.geo.census.gov'] or res['nominatim.openstreetmap.org']
    if not skip_geocode and not geo_ok:
        blocked_req += ['geocoding.geo.census.gov', 'nominatim.openstreetmap.org']
    return res, blocked_req


def step(cmd, code, label):
    print(f'\n== {label}: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f'RESULT: {json.dumps({"ok": False, "stage": label, "exit": r.returncode})}')
        sys.exit(code)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('address')
    ap.add_argument('--lat', type=float); ap.add_argument('--lon', type=float)
    ap.add_argument('--work', default=os.path.join(os.getcwd(), 'sites'))
    ap.add_argument('--skip-preflight', action='store_true')
    a = ap.parse_args()
    if (a.lat is None) != (a.lon is None):
        print('RESULT: ' + json.dumps({'ok': False, 'stage': 'input', 'error': '--lat and --lon go together'})); sys.exit(2)
    if not a.skip_preflight:
        res, blocked = preflight(a.lat is not None)
        print('network:', ', '.join(f"{h} {'ok' if v else 'BLOCKED'}" for h, v in res.items()))
        if blocked:
            print('RESULT: ' + json.dumps({'ok': False, 'stage': 'network', 'blocked_hosts': sorted(set(blocked)),
                                           'optional_blocked': [h for h, v in res.items() if not v and h not in blocked]}))
            sys.exit(3)
    os.makedirs(a.work, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'fetch'))
    from fetch_site import slugify
    slug = slugify(a.address); folder = os.path.join(os.path.abspath(a.work), slug)
    if os.path.exists(folder):
        k = 2
        while os.path.exists(os.path.join(os.path.abspath(a.work), f'run{k}', slug)): k += 1
        work = os.path.join(os.path.abspath(a.work), f'run{k}'); os.makedirs(work, exist_ok=True)
        folder = os.path.join(work, slug)
    else:
        work = os.path.abspath(a.work)
    py = sys.executable
    fetch = [py, os.path.join(ROOT, 'pipeline', 'fetch', 'fetch_site.py'), a.address, '--out-root', work]
    if a.lat is not None: fetch += ['--lat', str(a.lat), '--lon', str(a.lon)]
    print(f'\n== fetch: {" ".join(fetch)}', flush=True)
    if subprocess.run(fetch).returncode != 0:
        # USGS and Census services fail transiently (502/503); one clean retry in a fresh folder
        print('\n== fetch failed; retrying once in 20 s', flush=True)
        import time; time.sleep(20)
        k = 2
        while os.path.exists(os.path.join(os.path.abspath(a.work), f'retry{k}', slug)): k += 1
        work = os.path.join(os.path.abspath(a.work), f'retry{k}'); os.makedirs(work, exist_ok=True)
        folder = os.path.join(work, slug)
        fetch[fetch.index('--out-root') + 1] = work
        step(fetch, 4, 'fetch (retry)')
    print('\n== osm (streets and footprints; optional)', flush=True)
    subprocess.run([py, os.path.join(ROOT, 'pipeline', 'osm', 'fetch_osm.py'), folder])
    step([py, os.path.join(ROOT, 'pipeline', 'build', 'analyze_site.py'), folder], 5, 'analyze')
    step([py, os.path.join(ROOT, 'pipeline', 'build', 'build_model.py'), folder], 6, 'build')
    scene = json.load(open(os.path.join(folder, 'scene.json')))
    site = json.load(open(os.path.join(folder, 'site.json')))
    summary = dict(ok=True, html=os.path.join(folder, 'index.html'), folder=folder, address=site['address'],
                   matched_address=site.get('matched_address'), geocoder=(site.get('geocoder') or {}).get('name'),
                   lat=site['lat'], lon=site['lon'], street_source=scene['street']['source'],
                   datasets=[f"{d['name']} ({d['year']})" for d in scene['datasets']],
                   houses=[{k: h.get(k) for k in ('id', 'role', 'w', 'd', 'wall_h', 'ridge_h', 'stories')} | {'roof': h['roof']['type'], 'pitch': h['roof'].get('pitch_deg')} for h in scene['houses']],
                   trees=len(scene['trees']))
    print('RESULT: ' + json.dumps(summary))


if __name__ == '__main__':
    main()
