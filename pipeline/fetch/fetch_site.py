#!/usr/bin/env python3
"""Fetch geocoding, EPT LiDAR and a bare-earth grid into data/<address-slug>/."""
import argparse
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone

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

    def get(self, url):
        for attempt in range(1, 3):
            self.note(f'HTTP GET {url} attempt={attempt}')
            try:
                request = urllib.request.Request(url, headers={'User-Agent': self.user_agent})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    self.note(f'HTTP status={response.status} bytes={len(raw)} final_url={response.url}')
                    return raw
            except Exception as exc:
                self.note(f'HTTP failure: {type(exc).__name__}: {exc}')
                if attempt == 2:
                    raise
                time.sleep(1)

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


def decode_dem(raw):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'TIFF' or image.mode != 'F' or image.size != (COLS, ROWS):
            raise ValueError(f'Expected F32 TIFF {COLS}x{ROWS}; got {image.format} {image.mode} {image.size}')
        nodata = image.tag_v2.get(42113)
        if nodata is not None:
            nodata = float(str(nodata).strip('\x00'))
        values = [float(v) for v in image.getdata()]
    # Reject missing coverage rather than interpolating holes or accepting a rendered image.
    if any(not math.isfinite(v) or v <= -1e20 or v == nodata for v in values):
        raise ValueError('DEM contains NoData or non-finite cells')
    return values


def dem_sources(lat, lon):
    sources = []
    if 39.15 < lat < 39.8 and -77.75 < lon < -77.0:
        sources.append(FREDERICK)
    if 37.8 < lat < 39.8 and -79.6 < lon < -75.0:
        sources.append(MARYLAND)
    sources.append(USGS)
    return sources


def fetch_dem(site, output, fetch):
    bbox = terrain_bbox(site['lat'], site['mercator_center'])
    parameters = {
        'bbox': ','.join(f'{v:.9f}' for v in bbox), 'bboxSR': '3857', 'imageSR': '3857',
        'size': f'{COLS},{ROWS}', 'format': 'tiff', 'pixelType': 'F32',
        'interpolation': 'RSP_BilinearInterpolation', 'adjustAspectRatio': 'false',
        'compression': 'LZ77', 'noData': '-999999',
        'renderingRule': json.dumps({'rasterFunction': 'None'}, separators=(',', ':')),
        'f': 'image',
    }
    errors = []
    for source in dem_sources(site['lat'], site['lon']):
        url = source + '/exportImage?' + urllib.parse.urlencode(parameters)
        try:
            # Known sources below serve raw elevation in meters, NAVD88 in CONUS.
            # Keep the service's own metadata alongside the request for auditability.
            metadata = fetch.get_json(source + '?f=pjson')
            if metadata.get('bandCount') != 1:
                raise ValueError('Expected a single-band elevation service')
            raw = fetch.get(url)
            meters = decode_dem(raw)
            if any(v == -999999 for v in meters):
                raise ValueError('DEM contains requested NoData sentinel')
            # Outside CONUS the national service does not universally use NAVD88.
            if source == USGS and not (24 <= site['lat'] <= 50 and -125 <= site['lon'] <= -66):
                raise ValueError('NAVD88 not guaranteed outside CONUS; a datum-aware source is needed')
            (output / 'dem-source.tif').write_bytes(raw)
            write_json(output / 'dem-service.json', metadata)
            result = {
                'schema_version': 1, 'rows': ROWS, 'cols': COLS, 'cell_size_ft': CELL_FT,
                'units': 'feet', 'vertical_datum': 'NAVD88', 'horizontal_crs': 'EPSG:3857',
                'mercator_center': site['mercator_center'], 'bbox_3857': bbox,
                'row_order': 'north_to_south', 'column_order': 'west_to_east',
                'sample_location': 'cell_center', 'axes': {'x': 'east', 'z': 'south'},
                'elevations_ft': [v / METERS_PER_FOOT for v in meters],
                'source': {'url': source, 'export_url': url, 'parameters': parameters,
                           'elevation_units': 'meters', 'vertical_datum': 'NAVD88',
                           'meters_per_output_foot': METERS_PER_FOOT},
                'downloaded_utc': utc_now(),
            }
            write_json(output / 'dem.json', result)
            fetch.note(f'DEM saved: {ROWS * COLS} cells, source={source}')
            return
        except Exception as exc:
            errors.append(f'{source}: {exc}')
            fetch.note(f'DEM source failed: {errors[-1]}')
    raise ValueError('All DEM sources failed: ' + '; '.join(errors))


def run_lidar(site, output, fetch):
    command = [sys.executable, '-u', str(PROJECT / 'fetch_lidar.py'), site['address'],
               '--lat', str(site['lat']), '--lon', str(site['lon']),
               '--half', str(site['half_width_mercator']), '--out', str(output / 'lidar')]
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
    p.add_argument('--out-root', type=Path, default=PROJECT / 'data')
    p.add_argument('--user-agent', default='address-modeling-local-fetch/1.0',
                   help='Identifying HTTP User-Agent; may include your contact URL/email')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.lat is None) != (args.lon is None):
        p.error('--lat and --lon must be supplied together')
    if not math.isfinite(args.half) or args.half <= 0:
        p.error('--half must be finite and positive')
    try:
        slug = slugify(args.address)
        if args.lat is not None:
            validate_coordinates(args.lat, args.lon)
        # Check the sole dependency before making any network requests.
        from PIL import Image  # noqa: F401
    except (ValueError, ImportError) as exc:
        p.error(str(exc))
    output = args.out_root.resolve() / slug
    # An exclusive directory claim also prevents concurrent runs for the same address.
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        p.error(f'Output already exists: {output}. Preserve it; choose a new --out-root for another run.')
    with (output / 'fetch.log').open('w', encoding='utf-8') as log:
        fetch = Fetcher(log, args.user_agent)
        command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        fetch.note('COMMAND ' + (subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)))
        try:
            lat, lon, matched, geocoder = geocode(args.address, args.lat, args.lon, fetch)
            site = {'schema_version': 1, 'slug': slug, 'address': args.address,
                    'matched_address': matched, 'lat': lat, 'lon': lon,
                    'mercator_center': mercator(lat, lon), 'half_width_mercator': args.half,
                    'geocoder': geocoder, 'created_utc': utc_now(), 'status': 'fetching'}
            write_json(output / 'site.json', site)
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
            write_json(output / 'site.json', site)
            if failures:
                raise ValueError('; '.join(failures))
            fetch.note(f'Complete: {output}')
            fetch.note('CLI exit code: 0')
            return 0
        except Exception as exc:
            fetch.note(f'FAILED: {type(exc).__name__}: {exc}')
            fetch.note('CLI exit code: 1')
            return 1


if __name__ == '__main__':
    sys.exit(main())
