#!/usr/bin/env python3
"""
locate.py - find the house itself for a US street address, not just a point on the street.

    python locate.py "3625 Example Street, Town, ST 12345"   -> prints JSON {lat, lon, source, ...}

Order of sources:
  1. Census geocoder (then Nominatim) for an approximate, usually street-interpolated, point.
  2. U.S. DOT National Address Database (NAD) address points within 400 m (then 1200 m) of it,
     matched on house number and street name. These sit on the structure or parcel; public domain.
  3. If no geocoder matched: NAD by number + ZIP + street (slow attribute query).
  4. Otherwise the geocoder point as-is, flagged precision "street".
Standard library only.
"""
import json, math, re, sys, time, urllib.parse, urllib.request

NAD = ('https://services.arcgis.com/xOi1kZaI0eWDREZv/ArcGIS/rest/services/'
       'Address_Points_from_National_Address_Database_view/FeatureServer/0/query')
CENSUS = 'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress'
NOMINATIM = 'https://nominatim.openstreetmap.org/search'
UA = {'User-Agent': 'address-3d-model/1.0 (https://github.com/lagostimgomes/address-3d-model)'}
FIELDS = 'Add_Number,AddNum_Suf,StNam_Full,Unit,Post_City,State,Zip_Code,Placement,NAD_Source,DateUpdate'
SUFFIX = {'STREET': 'ST', 'AVENUE': 'AVE', 'BOULEVARD': 'BLVD', 'DRIVE': 'DR', 'ROAD': 'RD', 'LANE': 'LN', 'COURT': 'CT',
          'CIRCLE': 'CIR', 'PLACE': 'PL', 'TERRACE': 'TER', 'PARKWAY': 'PKWY', 'HIGHWAY': 'HWY', 'TRAIL': 'TRL', 'WAY': 'WAY',
          'NORTH': 'N', 'SOUTH': 'S', 'EAST': 'E', 'WEST': 'W', 'NORTHEAST': 'NE', 'NORTHWEST': 'NW', 'SOUTHEAST': 'SE', 'SOUTHWEST': 'SW'}


def get_json(url, params, timeout=30, tries=2):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url + '?' + urllib.parse.urlencode(params), headers=UA), timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e; time.sleep(2 * (i + 1))
    raise last


def parse(address):
    m = re.match(r'\s*(\d+)\s*([A-Za-z])?\s+([^,]+)', address)
    if not m: return None
    z = re.search(r'\b(\d{5})(?:-\d{4})?\s*$', address.strip())
    return dict(number=int(m.group(1)), suffix=(m.group(2) or '').upper(), street=m.group(3).strip(), zip=z.group(1) if z else None)


def norm(street):
    toks = re.sub(r'[^A-Z0-9 ]', ' ', (street or '').upper()).split()
    return [SUFFIX.get(t, t) for t in toks]


def street_score(a, b):
    A, B = norm(a), norm(b)
    if not A or not B: return 0.0
    core_a = [t for t in A if t not in SUFFIX.values()] or A
    core_b = [t for t in B if t not in SUFFIX.values()] or B
    s = len(set(core_a) & set(core_b)) / max(len(set(core_a)), 1)
    if set(A) & set(B) - set(core_a): s += 0.1           # suffix/direction agreement
    return s


def nad_query(where, near=None, radius=600):
    p = dict(where=where, outFields=FIELDS, returnGeometry='true', outSR=4326, f='json', resultRecordCount=50)
    if near:
        p.update(geometry=f'{near[1]},{near[0]}', geometryType='esriGeometryPoint', inSR=4326, distance=radius, units='esriSRUnit_Meter')
    r = get_json(NAD, p)
    if 'error' in r: raise RuntimeError(f"NAD error: {r['error']}")
    return [dict(a=f['attributes'], lat=f['geometry']['y'], lon=f['geometry']['x']) for f in r.get('features', [])]


def best_nad(feats, want, near=None):
    best = None
    for f in feats:
        a = f['a']
        if (a.get('AddNum_Suf') or '').upper() != want['suffix']: continue
        sc = street_score(want['street'], a.get('StNam_Full'))
        if sc < 0.5: continue
        units = 1 if not a.get('Unit') else 0                 # prefer the building point over unit points
        d = math.hypot((f['lat'] - near[0]) * 364000, (f['lon'] - near[1]) * 364000 * math.cos(math.radians(near[0]))) if near else 0
        key = (sc, units, -d)
        if best is None or key > best[0]: best = (key, f)
    return best[1] if best else None


def census(address):
    r = get_json(CENSUS, dict(address=address, benchmark='Public_AR_Current', format='json'))
    m = r.get('result', {}).get('addressMatches', [])
    if not m: return None
    c = m[0]['coordinates']; z = (m[0].get('addressComponents') or {}).get('zip')
    return dict(lat=float(c['y']), lon=float(c['x']), matched=m[0].get('matchedAddress'), zip=z)


def nominatim(address):
    r = get_json(NOMINATIM, dict(q=address, format='jsonv2', countrycodes='us', limit=1))
    if not r: return None
    return dict(lat=float(r[0]['lat']), lon=float(r[0]['lon']), matched=r[0].get('display_name'), zip=None)


def locate(address):
    want = parse(address); log = []
    approx = None
    for name, fn in (('census', census), ('nominatim', nominatim)):
        try:
            approx = fn(address)
            if approx: approx['source'] = name; break
            log.append(f'{name}: no match')
        except Exception as e: log.append(f'{name} failed: {e}')
    if want and approx:
        # fast: spatial search around the street-interpolated point
        for radius in (400, 1200):
            try:
                f = best_nad(nad_query(f"Add_Number={want['number']}", near=(approx['lat'], approx['lon']), radius=radius), want, (approx['lat'], approx['lon']))
                if f: return result(f, 'nad-near-' + approx['source'], log)
            except Exception as e: log.append(f'NAD near-search ({radius} m) failed: {e}'); break
        log.append('NAD: no match near the geocoded point')
    elif want and want['zip']:
        # slow fallback when no geocoder matched: attribute search by number and ZIP
        try:
            f = best_nad(nad_query(f"Add_Number={want['number']} AND Zip_Code='{want['zip']}'"), want)
            if f: return result(f, 'nad', log)
            log.append('NAD: no number+ZIP+street match')
        except Exception as e: log.append(f'NAD failed: {e}')
    if approx:
        return dict(ok=True, lat=approx['lat'], lon=approx['lon'], source=approx['source'], precision='street',
                    matched_address=approx.get('matched'), log=log)
    return dict(ok=False, error='address not found by Census, Nominatim or NAD', log=log)


def result(f, source, log):
    a = f['a']
    matched = ' '.join(str(x) for x in [a.get('Add_Number'), a.get('AddNum_Suf') or '', a.get('StNam_Full')] if x).replace('  ', ' ')
    matched += f", {a.get('Post_City')}, {a.get('State')} {a.get('Zip_Code')}"
    return dict(ok=True, lat=round(f['lat'], 7), lon=round(f['lon'], 7), source=source, precision='address-point',
                matched_address=matched, nad=dict(placement=a.get('Placement'), provider=a.get('NAD_Source'), updated=a.get('DateUpdate')), log=log)


if __name__ == '__main__':
    print(json.dumps(locate(' '.join(sys.argv[1:]))))
