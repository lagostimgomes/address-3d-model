#!/usr/bin/env python3
"""OSM evidence -> features.json, plus compatible roads.json/osm_buildings.json.

python pipeline/osm/fetch_osm.py <folder> [--raw-json cached-overpass.json]
No status-endpoint gate. Failures are explicit; missing mapping is never lawn.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fetch'))
from coverage import contains, distance, area

ENDPOINTS = ['https://overpass-api.de/api/interpreter',
             'https://overpass.kumi.systems/api/interpreter',
             'https://overpass.private.coffee/api/interpreter']
R = 6378137.0
FT = 1 / .3048


def to_model(lat, lon, site):
    x = math.radians(lon)*R
    y = R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
    cx, cy = site['mercator_center']
    k = math.cos(math.radians(site['lat']))*FT
    return [round((x-cx)*k,4),round(-(y-cy)*k,4)]


def query_for(site, radius_m=350, bounds_ft=None):
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('Query radius must be finite and positive')
    if bounds_ft:
        radius_m = max(radius_m, max(math.hypot(x,z) for x in bounds_ft[:2] for z in bounds_ft[2:])*.3048+30)
    lat, lon = site['lat'],site['lon']
    a = f'(around:{radius_m:.2f},{lat:.8f},{lon:.8f})'
    # One key-regex selection avoids a separate spatial scan for every key.
    tag='[~"^(building|building:part|highway|landuse|landcover|leisure|water|area:highway)$"~"."]'
    statements=[f'way{tag}{a};',f'relation["type"="multipolygon"]{tag}{a};']
    for tag in ('["amenity"="parking"]','["natural"="water"]','["service"="driveway"]'):
        statements += [f'way{tag}{a};',f'relation["type"="multipolygon"]{tag}{a};']
    statements.append(f'node["entrance"]{a};')
    return '[out:json][timeout:20];('+''.join(statements)+');out meta geom;'


def overpass(query, log):
    last = None
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(url+'?'+urllib.parse.urlencode({'data':query}),
                                         headers={'User-Agent':'address-3d-model/2.0'})
            with urllib.request.urlopen(req,timeout=25) as response:
                raw = response.read()
            result = json.loads(raw)
            if result.get('remark') or not isinstance(result.get('elements'),list):
                raise ValueError('Incomplete Overpass response: '+str(result.get('remark','missing elements')))
            log.append(f'GET {url} -> {len(raw)} bytes')
            return result,url
        except Exception as exc:
            last = exc
            log.append(f'GET {url} failed: {type(exc).__name__}: {exc}')
    raise RuntimeError(f'All Overpass interpreters failed: {last}')


def closed_rings(parts):
    """Join unordered/reversed member ways; never invent a missing edge."""
    pending = [list(p) for p in parts if len(p)>=2]
    rings, incomplete = [], any(len(p)<2 for p in parts)
    while pending:
        ring = pending.pop(0)
        while ring[0] != ring[-1]:
            found = False
            for i,p in enumerate(pending):
                if ring[-1] == p[0]: ring += p[1:]
                elif ring[-1] == p[-1]: ring += p[-2::-1]
                elif ring[0] == p[-1]: ring = p[:-1]+ring
                elif ring[0] == p[0]: ring = p[:0:-1]+ring
                else: continue
                pending.pop(i)
                found = True
                break
            if not found:
                incomplete = True
                break
        if ring[0] == ring[-1] and len(set(ring))>=3:
            rings.append(ring)
        else:
            incomplete = True
    return rings,incomplete


def geometry_rings(el, site):
    def pts(g):
        if any(not p or 'lat' not in p or 'lon' not in p for p in g): return []
        return [tuple(to_model(p['lat'],p['lon'],site)) for p in g]
    if el['type']=='way':
        p = pts(el.get('geometry',[]))
        return ([p] if len(set(p))>=3 and p[0]==p[-1] else []),[],False
    outer,inner = [],[]
    for m in el.get('members',[]):
        if m.get('type')!='way': continue
        role = m.get('role','outer') or 'outer'
        if role not in ('outer','inner'): continue
        (inner if role=='inner' else outer).append(pts(m.get('geometry',[])))
    ou,ob = closed_rings(outer)
    inn,ib = closed_rings(inner)
    return ou,inn,ob or ib or not ou


def width_feet(value):
    if not value: return None
    s = str(value).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|meters?|metres?|ft|feet|foot|')?",s)
    if m:
        v = float(m[1])*(1 if m[2] in ('ft','feet','foot',"'") else FT)
        return round(v,3) if 0<v<=500 else None
    m = re.fullmatch(r'''(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*"''',s)
    if m:
        v = int(m[1])+float(m[2])/12
        return round(v,3) if 0<v<=500 else None
    return None


def provenance(el):
    return dict(source='osm',source_id=f"{el['type']}/{el['id']}",
                source_date=None,edited_utc=el.get('timestamp'),confidence=.65,
                confidence_components=dict(geometry='mapped_unverified',temporal='acquisition_unknown',
                                           completeness='point' if el['type']=='node' else 'mapped_geometry',method='osm'))


def surface_kind(t):
    if t.get('natural')=='water' or 'water' in t: return 'water'
    if t.get('amenity')=='parking': return 'parking'
    if t.get('footway')=='sidewalk' or t.get('area:highway') in ('footway','pedestrian'): return 'sidewalk'
    if t.get('service')=='driveway': return 'driveway'
    if t.get('highway')=='pedestrian': return 'plaza'
    if 'area:highway' in t or (t.get('area')=='yes' and 'highway' in t): return 'road_area'
    if t.get('landuse') in ('grass','meadow') or t.get('landcover')=='grass': return 'lawn'
    if t.get('surface') in ('asphalt','concrete','paving_stones'): return 'other_paved'
    return 'other'


def normalize(data,site):
    result = dict(schema_version=1,status='ok',buildings=[],surfaces=[],ways=[],entrances=[],
                  notes=[],downloaded_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  license='ODbL; (c) OpenStreetMap contributors')
    elems = data.get('elements',[])
    consumed = {'building':set(),'surface':set()}
    for el in elems:
        if el.get('type')!='relation': continue
        outer,inner,bad = geometry_rings(el,site)
        if outer and not bad:
            family='building' if el.get('tags',{}).get('building','no')!='no' else 'surface'
            consumed[family].update(m['ref'] for m in el.get('members',[]) if m.get('type')=='way')
    seen = set()
    for el in elems:
        key = (el.get('type'),el.get('id'))
        if key in seen: continue
        seen.add(key)
        t = el.get('tags',{})
        prov = provenance(el)
        if el['type']=='node':
            if t.get('entrance') and t['entrance']!='no' and 'lat' in el and 'lon' in el:
                x,z = to_model(el['lat'],el['lon'],site)
                result['entrances'].append(dict(id=prov['source_id'],x=x,z=z,building_id=None,tags=t,**prov))
            continue
        if el['type']=='way' and 'highway' in t and t.get('area')!='yes':
            g=el.get('geometry',[])
            if len(g)>=2 and all(p and 'lat' in p and 'lon' in p for p in g):
                w=width_feet(t.get('width'))
                result['ways'].append(dict(id=prov['source_id'],kind=t['highway'],
                    centerline=[to_model(p['lat'],p['lon'],site) for p in g],width_ft=w,
                    width_source='tag' if w is not None else None,
                    **{k:t.get(k) for k in ('lanes','sidewalk','layer','bridge','tunnel','name')},tags=t,**prov))
        family='building' if t.get('building','no')!='no' else 'surface'
        if el['type']=='way' and el['id'] in consumed[family]: continue
        outer,inner,bad=geometry_rings(el,site)
        if bad:
            result['notes'].append(f"{prov['source_id']}: incomplete multipolygon skipped")
            if el.get('bounds'):
                b=el['bounds']
                nw=to_model(b['maxlat'],b['minlon'],site)
                se=to_model(b['minlat'],b['maxlon'],site)
                result.setdefault('unresolved_geometries',[]).append(dict(id=prov['source_id'],
                    bounds_ft=[nw[0],se[0],nw[1],se[1]],tags=t))
            continue
        for i,ring in enumerate(outer):
            feature=dict(id=prov['source_id']+(f':{i}' if len(outer)>1 else ''),
                         polygon=[list(p) for p in ring],holes=[],tags=t,**prov)
            feature['holes']=[[list(p) for p in h] for h in inner if contains(h[0],feature)]
            if t.get('building','no')!='no':
                result['buildings'].append(feature)
            elif t.get('building:part','no')!='no':
                result.setdefault('building_parts',[]).append(feature)
            else:
                feature['kind']=surface_kind(t)
                result['surfaces'].append(feature)
    for ent in result['entrances']:
        candidates=[(distance([ent['x'],ent['z']],b),area(b),b['id']) for b in result['buildings']]
        if candidates:
            d,_,bid=min(candidates)
            if d<=6: ent['building_id']=bid
    return result


def write_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=1,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def fetch_features(site,folder,radius_m=350,raw_json=None):
    folder=Path(folder)
    query=query_for(site,radius_m,site.get('bounds_ft'))
    log=[f'query: {query}']
    try:
        if raw_json:
            data=json.loads(Path(raw_json).read_text(encoding='utf-8'))
            endpoint='cached-input'
            if data.get('remark') or not isinstance(data.get('elements'),list):
                raise ValueError('Cached Overpass response is incomplete')
        else:
            data,endpoint=overpass(query,log)
        result=normalize(data,site)
        write_json(folder/'osm-raw.json',data)
        result.update(endpoint=endpoint,query=query,log=log)
    except Exception as exc:
        result=dict(schema_version=1,status='unavailable',error=str(exc),
                    buildings=[],surfaces=[],ways=[],entrances=[],notes=[str(exc)],log=log)
    prior=folder/'features.json'
    if prior.exists():
        old=json.loads(prior.read_text(encoding='utf-8'))
        if result['status']=='unavailable' and old.get('buildings'):
            error=result['error']
            result=dict(old,status='stale',error=error,refresh_log=log,
                        notes=old.get('notes',[])+['Refresh failed; retaining prior mapped evidence: '+error])
        if old.get('coverage'): result['coverage']=old['coverage']
    write_json(prior,result)
    meta={k:result[k] for k in ('status','log','endpoint','query','error') if k in result}
    meta.update(source='OpenStreetMap via Overpass',license='ODbL; (c) OpenStreetMap contributors')
    write_json(folder/'roads.json',dict(meta,roads=[dict(id=w['source_id'],highway=w['kind'],
               name=w.get('name'),points=w['centerline'],tags=w['tags'],width_ft=w['width_ft']) for w in result['ways']]))
    write_json(folder/'osm_buildings.json',dict(meta,buildings=[dict(id=b['source_id'],tags=b['tags'],
               points=b['polygon'],holes=b['holes']) for b in result['buildings']]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder',type=Path)
    ap.add_argument('--radius-m',type=float,default=350)
    ap.add_argument('--raw-json',type=Path)
    ap.add_argument('--roads',type=float,default=0)
    ap.add_argument('--buildings',type=float,default=0)
    args=ap.parse_args()
    if any(not math.isfinite(v) or v<0 for v in (args.radius_m,args.roads,args.buildings)) or args.radius_m==0:
        ap.error('Radii must be finite and nonnegative; --radius-m must be positive')
    site=json.loads((args.folder/'site.json').read_text(encoding='utf-8'))
    f=fetch_features(site,args.folder,max(args.radius_m,args.roads,args.buildings),args.raw_json)
    print(f"OSM {f['status']}: {len(f['buildings'])} buildings, {len(f['surfaces'])} surfaces, "
          f"{len(f['ways'])} ways, {len(f['entrances'])} entrances")


if __name__=='__main__': main()
