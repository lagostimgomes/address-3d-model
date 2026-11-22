#!/usr/bin/env python3
"""Fetch geocoding, EPT LiDAR and a bare-earth grid into data/<address-slug>/."""
import argparse
import io
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata
import uuid
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from coverage import grid_spec, plan

PROJECT = Path(__file__).resolve().parents[2]
CENSUS = 'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress'
NOMINATIM = 'https://nominatim.openstreetmap.org/search'
FREDERICK = 'https://mdgeodata.md.gov/lidar/rest/services/Frederick/MD_frederick_dem_m/ImageServer'
MARYLAND = 'https://mdgeodata.md.gov/lidar/rest/services/Statewide/MD_statewide_dem_m/ImageServer'
USGS = 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer'
ROWS = COLS = 120
CELL_FT = 5.2868
METERS_PER_FOOT = 0.3048


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def slugify(address):
    ascii_text = unicodedata.normalize('NFKD', address).encode('ascii', 'ignore').decode()
    slug = re.sub(r'[^a-z0-9]+', '-', ascii_text.lower()).strip('-')
    if not slug:
        raise ValueError('Address must contain letters or numbers for the output slug')
    return slug


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class Fetcher:
    def __init__(self, log, user_agent, timeout=30):
        self.log = log
        self.user_agent = user_agent
        self.timeout = timeout

    def note(self, message):
        line = f'[{utc_now()}] {message}'
        print(line, flush=True)
        self.log.write(line + '\n')
        self.log.flush()

    def get(self, url, attempts=2, backoff=(1,)):
        for attempt in range(1, attempts+1):
            self.note(f'HTTP GET {url} attempt={attempt}')
            try:
                request = urllib.request.Request(url, headers={'User-Agent': self.user_agent})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    self.note(f'HTTP status={response.status} bytes={len(raw)} final_url={response.url}')
                    return raw
            except Exception as exc:
                self.note(f'HTTP failure: {type(exc).__name__}: {exc}')
                if attempt == attempts or (isinstance(exc, urllib.error.HTTPError) and exc.code not in (408,429,500,502,503,504)):
                    raise
                time.sleep(backoff[min(attempt-1,len(backoff)-1)])

    def get_json(self, url):
        value = json.loads(self.get(url))
        if isinstance(value, dict) and 'error' in value:
            raise ValueError(f'ArcGIS error: {value["error"]}')
        return value


def geocode(address, lat, lon, fetch):
    if lat is not None:
        return lat, lon, address, {'name': 'coordinates', 'source': 'CLI --lat/--lon'}
    sources = [
        ('census', CENSUS, {'address': address, 'benchmark': 'Public_AR_Current', 'format': 'json'}),
        ('nominatim', NOMINATIM, {'q': address, 'format': 'jsonv2', 'countrycodes': 'us', 'limit': 1}),
    ]
    for name, base, parameters in sources:
        url = base + '?' + urllib.parse.urlencode(parameters)
        try:
            response = fetch.get_json(url)
            if name == 'census':
                match = response['result']['addressMatches'][0]
                lat, lon = float(match['coordinates']['y']), float(match['coordinates']['x'])
                matched = match['matchedAddress']
            else:
                lat, lon = float(response[0]['lat']), float(response[0]['lon'])
                matched = response[0]['display_name']
            validate_coordinates(lat, lon)
            return lat, lon, matched, {'name': name, 'url': url, 'parameters': parameters}
        except Exception as exc:
            fetch.note(f'{name} geocoding failed: {exc}')
    raise ValueError('Both geocoders failed; retry with --lat LAT --lon LON')


def validate_coordinates(lat, lon):
    if not (math.isfinite(lat) and math.isfinite(lon) and -85 < lat < 85 and -180 <= lon <= 180):
        raise ValueError('Coordinates must be finite, -85 < latitude < 85, -180 <= longitude <= 180')


def mercator(lat, lon):
    return [6378137 * math.radians(lon),
            6378137 * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))]


def terrain_bbox(lat, center):
    # Convert ground feet to Web Mercator units at the site's latitude.
    half = COLS * CELL_FT * METERS_PER_FOOT / (2 * math.cos(math.radians(lat)))
    x, y = center
    return [x - half, y - half, x + half, y + half]


def decode_dem(raw, cols=COLS, rows=ROWS):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'TIFF' or image.mode != 'F' or image.size != (cols, rows):
            raise ValueError(f'Expected F32 TIFF {cols}x{rows}; got {image.format} {image.mode} {image.size}')
        nodata = image.tag_v2.get(42113)
        if nodata is not None:
            nodata = float(str(nodata).strip('\x00'))
        values = [float(v) for v in (image.get_flattened_data() if hasattr(image,'get_flattened_data') else image.getdata())]
    # Reject missing coverage rather than interpolating holes or accepting a rendered image.
    if any(not math.isfinite(v) or v <= -1e20 or v == nodata for v in values):
        raise ValueError('DEM contains NoData or non-finite cells')
    return values


def dem_sources(lat, lon, state=None, county=None):
    # State/county must be resolved by a polygon-backed geography lookup.
    sources = []
    if state == '24':
        if county == '021': sources.append(FREDERICK)
        sources.append(MARYLAND)
    return sources + [USGS]


def fetch_dem(site, output, fetch):
    from dem_grid import fetch_dem as fetch_grid
    return fetch_grid(site, output, fetch, sys.modules[__name__])


def run_lidar(site, output, fetch):
    command = [sys.executable, '-u', str(PROJECT / 'fetch_lidar.py'), site['address'],
               '--lat', str(site['lat']), '--lon', str(site['lon']),
               '--half', str(site['half_width_mercator']), '--out', str(output / 'lidar')]
    if site.get('bounds_ft'):
        command = [sys.executable, '-u', str(Path(__file__).with_name('lidar_extent.py')), str(output)]
    fetch.note('COMMAND ' + (subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)))
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace') as process:
        for line in process.stdout:
            fetch.note('lidar: ' + line.rstrip())
        code = process.wait()
    fetch.note(f'LiDAR exit code: {code}')
    if code:
        raise ValueError(f'LiDAR downloader exited with {code}')
    manifest = json.loads((output / 'lidar' / 'manifest.json').read_text(encoding='utf-8'))
    datasets = manifest.get('datasets', {})
    if not datasets or any('error' in d or not d.get('nodes') for d in datasets.values()):
        raise ValueError('LiDAR manifest has failed or empty datasets; see fetch.log and manifest.json')
    for name, dataset in datasets.items():
        tiles = [output / 'lidar' / name / 'ept-data' / (node + '.laz') for node in dataset['nodes']]
        if sum(tile.stat().st_size for tile in tiles) != dataset['bytes']:
            raise ValueError(f'LiDAR byte count mismatch for {name}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('address')
    p.add_argument('--lat', type=float)
    p.add_argument('--lon', type=float)
    p.add_argument('--half', type=float, default=125, help='LiDAR half-width in Web Mercator units')
    p.add_argument('--bounds-ft', nargs=4, type=float, metavar=('XMIN','XMAX','ZMIN','ZMAX'),
                   help='Minimum requested cell edges in model feet; exclusive new output folder')
    p.add_argument('--expand-into', type=Path,
                   help='Stage one coverage expansion, preserve prior data in fetch-history, then promote on success')
    p.add_argument('--reuse-lidar-from', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--expansions', type=int, choices=(0,1), default=0)
    p.add_argument('--no-features', action='store_true', help='Legacy 120x120 acquisition without mapped preselection')
    p.add_argument('--osm-raw-json', type=Path, help='Replay cached Overpass response for reproducibility')
    p.add_argument('--out-root', type=Path, default=PROJECT / 'data')
    p.add_argument('--user-agent', default='address-modeling-local-fetch/1.0',
                   help='Identifying HTTP User-Agent; may include your contact URL/email')
    return p


def expand_existing(args):
    folder=args.expand_into.resolve()
    if not args.bounds_ft:
        raise ValueError('--expand-into requires --bounds-ft')
    old=json.loads((folder/'site.json').read_text(encoding='utf-8'))
    if old.get('status')!='complete' or old.get('slug')!=slugify(args.address):
        raise ValueError('Expansion requires a complete matching site; original data preserved')
    if args.lat is not None and (abs(old['lat']-args.lat)>1e-9 or abs(old['lon']-args.lon)>1e-9):
        raise ValueError('Expansion must preserve the original model origin')
    previous=json.loads((folder/'features.json').read_text(encoding='utf-8')) if (folder/'features.json').exists() else {}
    if (previous.get('coverage') or {}).get('expansions',0)>=1:
        raise ValueError('One expansion already attempted; keep explicit incomplete coverage')
    old_bounds=old.get('bounds_ft') or grid_spec()['bounds_ft']
    request=[min(old_bounds[0],args.bounds_ft[0]),max(old_bounds[1],args.bounds_ft[1]),
             min(old_bounds[2],args.bounds_ft[2]),max(old_bounds[3],args.bounds_ft[3])]
    lock=folder/'.fetch-expand.lock'
    with lock.open('x',encoding='utf-8') as handle:
        handle.write(utc_now())
    try:
        # Keep temporary/history paths short: long dataset names plus .bytes
        # sidecars can otherwise exceed Windows MAX_PATH during cache copying.
        tag=uuid.uuid4().hex[:12]
        stage_root=folder.parent/('.fx-'+tag)
        command=[args.address,'--lat',str(old['lat']),'--lon',str(old['lon']),
                 '--bounds-ft',*[str(v) for v in request],'--expansions','1',
                 '--out-root',str(stage_root),'--user-agent',args.user_agent,
                 '--reuse-lidar-from',str(folder/'lidar')]
        if args.osm_raw_json: command+=['--osm-raw-json',str(args.osm_raw_json.resolve())]
        code=main(command)
        staged=stage_root/old['slug']
        if code:
            write_json(folder/'expansion-failure.json',dict(status='failed',stage=str(staged),requested_bounds_ft=request))
            return code
        new=json.loads((staged/'site.json').read_text(encoding='utf-8'))
        b=new['bounds_ft']
        if b[0]>old_bounds[0]+1e-6 or b[1]<old_bounds[1]-1e-6 or b[2]>old_bounds[2]+1e-6 or b[3]<old_bounds[3]-1e-6:
            write_json(folder/'expansion-failure.json',dict(status='budget_prevents_monotone_expansion',stage=str(staged)))
            return 1
        # Backup by rename, then promote generated inputs; keep facts/locate/scene/HTML
        # untouched. site.json is the last commit marker. Roll back on local errors.
        history=folder/'fetch-history'/tag
        history.mkdir(parents=True)
        names=[p.name for p in staged.iterdir() if p.name!='site.json']+['site.json']
        promoted=[]
        shutil.copy2(folder/'site.json',history/'site.json')
        backed=['site.json']
        try:
            # A crash during multi-file promotion must fail closed for the builder.
            write_json(folder/'site.json',dict(old,status='expanding',expansion_history=str(history)))
            for name in names:
                target=folder/name
                if target.exists() and name!='site.json':
                    target.replace(history/name)
                    backed.append(name)
                (staged/name).replace(target)
                promoted.append(name)
        except Exception:
            failed=stage_root/'promotion-failed'
            failed.mkdir(exist_ok=True)
            for name in reversed(promoted): (folder/name).replace(failed/name)
            for name in reversed(backed): (history/name).replace(folder/name)
            raise
        print(f'Coverage expansion promoted; prior inputs preserved in {history}',flush=True)
        return 0
    finally:
        lock.unlink()


def main(argv=None):
    p = parser()
    tokens=list(sys.argv[1:] if argv is None else argv)
    # Accept Claude's run.py comma form, including a negative first bound, as
    # well as four separate numbers. Normalize before argparse option matching.
    if '--bounds-ft' in tokens:
        k=tokens.index('--bounds-ft')+1
        if k<len(tokens) and ',' in tokens[k]: tokens[k:k+1]=tokens[k].split(',')
    args = p.parse_args(tokens)
    if (args.lat is None) != (args.lon is None):
        p.error('--lat and --lon must be supplied together')
    if not math.isfinite(args.half) or args.half <= 0:
        p.error('--half must be finite and positive')
    try:
        slug = slugify(args.address)
        if args.bounds_ft is not None:
            grid_spec(args.bounds_ft)
        if args.lat is not None:
            validate_coordinates(args.lat, args.lon)
        # Check the sole dependency before making any network requests.
        from PIL import Image  # noqa: F401
    except (ValueError, ImportError) as exc:
        p.error(str(exc))
    if args.expand_into:
        try: return expand_existing(args)
        except Exception as exc:
            print(f'Coverage expansion failed; prior site preserved: {type(exc).__name__}: {exc}',flush=True)
            return 1
    output = args.out_root.resolve() / slug
    # An exclusive directory claim also prevents concurrent runs for the same address.
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        p.error(f'Output already exists: {output}. Preserve it; choose a new --out-root for another run.')
    if args.reuse_lidar_from and args.reuse_lidar_from.is_dir():
        shutil.copytree(args.reuse_lidar_from,output/'lidar')
        prior_features=args.reuse_lidar_from.parent/'features.json'
        if prior_features.exists(): shutil.copy2(prior_features,output/'features.json')
    with (output / 'fetch.log').open('w', encoding='utf-8') as log:
        fetch = Fetcher(log, args.user_agent)
        command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        fetch.note('COMMAND ' + (subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)))
        site = None
        try:
            lat, lon, matched, geocoder = geocode(args.address, args.lat, args.lon, fetch)
            site = {'schema_version': 1, 'slug': slug, 'address': args.address,
                    'matched_address': matched, 'lat': lat, 'lon': lon,
                    'mercator_center': mercator(lat, lon), 'half_width_mercator': args.half,
                    'geocoder': geocoder, 'created_utc': utc_now(), 'status': 'fetching'}
            write_json(output / 'site.json', site)
            features = None
            if not args.no_features:
                sys.path.insert(0, str(PROJECT/'pipeline'/'osm'))
                from fetch_osm import fetch_features
                if args.bounds_ft:
                    site['bounds_ft'] = grid_spec(args.bounds_ft)['bounds_ft']
                features = fetch_features(site, output, raw_json=args.osm_raw_json)
                spec, cov = plan(features, args.bounds_ft, args.expansions)
                site.update(bounds_ft=spec['bounds_ft'], rows=spec['rows'], cols=spec['cols'])
                features['coverage'] = cov
                write_json(output/'features.json', features)
                write_json(output/'site.json', site)
            elif args.bounds_ft is not None:
                site.update(grid_spec(args.bounds_ft))
                write_json(output/'site.json', site)
            failures = []
            for name, step in [('dem', fetch_dem), ('lidar', run_lidar)]:
                try:
                    step(site, output, fetch)
                except Exception as exc:
                    failures.append(f'{name}: {exc}')
                    fetch.note(f'STEP FAILED: {failures[-1]}')
            site['status'] = 'failed' if failures else 'complete'
            site['errors'] = failures
            site['finished_utc'] = utc_now()
            if features is not None:
                cov = features['coverage']
                for name, filename, key in [('dem','dem.json','dem_bounds_ft'),('lidar','lidar/manifest.json','lidar_bounds_ft')]:
                    path = output/filename
                    if path.exists() and not any(f.startswith(name+':') for f in failures):
                        cov[key] = json.loads(path.read_text(encoding='utf-8')).get('bounds_ft')
                if cov['dem_bounds_ft'] is None or cov['lidar_bounds_ft'] is None:
                    cov['status'] = 'incomplete'
                write_json(output/'features.json', features)
            write_json(output / 'site.json', site)
            if failures:
                raise ValueError('; '.join(failures))
            fetch.note(f'Complete: {output}')
            fetch.note('CLI exit code: 0')
            return 0
        except Exception as exc:
            if site is not None and site.get('status') != 'failed':
                site.update(status='failed', errors=[str(exc)], finished_utc=utc_now())
                write_json(output/'site.json', site)
            fetch.note(f'FAILED: {type(exc).__name__}: {exc}')
            fetch.note('CLI exit code: 1')
            return 1


if __name__ == '__main__':
    sys.exit(main())
