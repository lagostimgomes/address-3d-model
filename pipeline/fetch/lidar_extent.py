#!/usr/bin/env python3
"""Rectangular EPT acquisition without changing the original downloader.

Reads site.json bounds_ft; tests ALL dataset polygons intersecting the extent,
not only the center point. Caches existing valid tiles for one expansion pass.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import zipfile

from coverage import polygon_distance
from dem_grid import bbox_mercator

ROOT=Path(__file__).resolve().parents[2]


def load_downloader():
    spec=importlib.util.spec_from_file_location('original_lidar_downloader',ROOT/'fetch_lidar.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def intersects(geometry,box):
    # Geographic lon/lat ring distances are used only to test intersection == 0.
    west,south,east,north=box
    rect=dict(polygon=[[west,south],[east,south],[east,north],[west,north],[west,south]])
    polys=geometry['coordinates'] if geometry['type']=='MultiPolygon' else [geometry['coordinates']]
    return any(polygon_distance(rect,dict(polygon=p[0],holes=p[1:]))<1e-12 for p in polys)


def fetch_dataset(api,url,box,out):
    base=url.rsplit('/',1)[0]
    (out/'ept-hierarchy').mkdir(parents=True,exist_ok=True)
    (out/'ept-data').mkdir(parents=True,exist_ok=True)
    raw=api.get(url)
    (out/'ept.json').write_bytes(raw)
    ept=json.loads(raw)
    if str(ept.get('srs',{}).get('horizontal','3857'))!='3857':
        raise ValueError('EPT horizontal CRS must be EPSG:3857')
    wanted,queue,seen={},['0-0-0-0'],set()
    while queue:
        hk=queue.pop()
        if hk in seen: continue
        seen.add(hk)
        raw=api.get(f'{base}/ept-hierarchy/{hk}.json')
        (out/'ept-hierarchy'/f'{hk}.json').write_bytes(raw)
        for key,count in json.loads(raw).items():
            if not api.overlaps(api.node_box(key,ept['bounds']),box): continue
            if count==-1: queue.append(key)
            elif count>0: wanted[key]=count
    if not wanted: raise ValueError('Dataset intersects extent but has no point tiles there')
    total=0
    for i,(key,count) in enumerate(sorted(wanted.items()),1):
        dest=out/'ept-data'/f'{key}.laz'
        # A complete cache entry is recorded in a separate sidecar only after the
        # atomic rename; partial downloads and arbitrary existing files are not reused.
        stamp=dest.with_suffix('.bytes')
        cached=dest.exists() and stamp.exists() and dest.stat().st_size==int(stamp.read_text())
        if not cached:
            raw=api.get(f'{base}/ept-data/{key}.laz')
            if not raw.startswith(b'LASF'): raise ValueError('Tile does not have LASF header')
            tmp=dest.with_suffix('.partial')
            tmp.write_bytes(raw)
            tmp.replace(dest)
            stamp.write_text(str(len(raw)))
        total+=dest.stat().st_size
        print(f'  [{i}/{len(wanted)}] {key}.laz {count:,} pts'+(' cached' if cached else ''),flush=True)
    return dict(ept_url=url,nodes=sorted(wanted),points=sum(wanted.values()),bytes=total)


def main(folder):
    folder=Path(folder)
    site=json.loads((folder/'site.json').read_text(encoding='utf-8'))
    api=load_downloader()
    box=bbox_mercator(site,site['bounds_ft'])
    west,south,east,north=box
    ll=[math.degrees(west/6378137),math.degrees(2*math.atan(math.exp(south/6378137))-math.pi/2),
        math.degrees(east/6378137),math.degrees(2*math.atan(math.exp(north/6378137))-math.pi/2)]
    index=api.get_json(api.INDEX_URL)
    hits=[f['properties'] for f in index['features'] if intersects(f['geometry'],ll)]
    if not hits: raise ValueError('No USGS 3DEP point-cloud dataset intersects this extent')
    lidar=folder/'lidar'
    lidar.mkdir(exist_ok=True)
    manifest=dict(address_input=site['address'],matched_address=site.get('matched_address'),
                  lat=site['lat'],lon=site['lon'],mercator_center=site['mercator_center'],
                  bounds_ft=site['bounds_ft'],bbox_3857=box,
                  downloaded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),datasets={})
    for h in hits:
        if not h['name'] or h['name'] in ('.','..') or any(c in h['name'] for c in '/\\:'):
            raise ValueError('Unsafe dataset path component in EPT index')
        print(f"Downloading extent tiles: {h['name']}",flush=True)
        try:
            manifest['datasets'][h['name']]=fetch_dataset(api,h['url'],box,lidar/h['name'])
        except Exception as exc:
            manifest['datasets'][h['name']]=dict(ept_url=h['url'],error=str(exc))
            print(f'FAILED: {exc}',flush=True)
    (lidar/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(folder/'lidar.zip','w',zipfile.ZIP_STORED) as archive:
        for p in sorted(lidar.rglob('*')):
            if p.is_file() and p.suffix!='.partial': archive.write(p,p.relative_to(folder))
    if any('error' in d for d in manifest['datasets'].values()):
        raise ValueError('One or more intersecting EPT datasets failed; see manifest.json')


if __name__=='__main__': main(sys.argv[1])
