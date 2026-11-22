"""Raw F32 DEM requests with jurisdiction checks, backoff, and tiled fallback."""
import io
import json
import math
import urllib.parse

from coverage import grid_spec

GEOGRAPHIES = 'https://geocoding.geo.census.gov/geocoder/geographies/coordinates'


def jurisdiction(site, fetch):
    params=dict(x=site['lon'],y=site['lat'],benchmark='Public_AR_Current',
                vintage='Current_Current',format='json')
    url=GEOGRAPHIES+'?'+urllib.parse.urlencode(params)
    try:
        g=fetch.get_json(url).get('result',{}).get('geographies',{})
        states=g.get('States',[])
        counties=g.get('Counties',[])
        return dict(state=states[0].get('STATE') if len(states)==1 else None,
                    county=counties[0].get('COUNTY') if len(counties)==1 else None,url=url)
    except Exception as exc:
        fetch.note(f'Jurisdiction unresolved; using national DEM only: {exc}')
        return dict(state=None,county=None,url=url,error=str(exc))


def bbox_mercator(site,bounds):
    k=.3048/math.cos(math.radians(site['lat']))
    cx,cy=site['mercator_center']
    xmin,xmax,zmin,zmax=bounds
    return [cx+xmin*k,cy-zmax*k,cx+xmax*k,cy-zmin*k]


def parameters(bbox,cols,rows):
    return dict(bbox=','.join(f'{v:.9f}' for v in bbox),bboxSR='3857',imageSR='3857',
                size=f'{cols},{rows}',format='tiff',pixelType='F32',
                interpolation='RSP_BilinearInterpolation',adjustAspectRatio='false',
                compression='LZ77',noData='-999999',
                renderingRule='{"rasterFunction":"None"}',f='image')


def export(source,bbox,cols,rows,fetch,decode,usgs=False):
    p=parameters(bbox,cols,rows)
    url=source+'/exportImage?'+urllib.parse.urlencode(p)
    raw=fetch.get(url,attempts=4 if usgs else 2,backoff=(5,15,30) if usgs else (1,))
    values=decode(raw,cols=cols,rows=rows)
    if any(v==-999999 for v in values):
        raise ValueError('DEM contains requested NoData sentinel')
    return values,raw,dict(url=url,parameters=p)


def export_tiled(source,bbox,cols,rows,fetch,decode,output):
    """Split at exact pixel edges, including odd sizes; concatenate north to south."""
    from PIL import Image
    values=[None]*(rows*cols)
    audit=[]
    xs,ys=[0,cols//2,cols],[0,rows//2,rows]
    xmin,ymin,xmax,ymax=bbox
    for r in range(2):
        for c in range(2):
            x0,x1=xs[c:c+2]
            y0,y1=ys[r:r+2]
            box=[xmin+(xmax-xmin)*x0/cols,ymax-(ymax-ymin)*y1/rows,
                 xmin+(xmax-xmin)*x1/cols,ymax-(ymax-ymin)*y0/rows]
            p=parameters(box,x1-x0,y1-y0)
            url=source+'/exportImage?'+urllib.parse.urlencode(p)
            raw=fetch.get(url,attempts=2,backoff=(5,))
            v=decode(raw,cols=x1-x0,rows=y1-y0)
            if any(a==-999999 for a in v): raise ValueError('DEM tile contains requested NoData sentinel')
            name=f'dem-source-r{r}-c{c}.tif'
            (output/name).write_bytes(raw)
            for j in range(y1-y0):
                values[(y0+j)*cols+x0:(y0+j)*cols+x1]=v[j*(x1-x0):(j+1)*(x1-x0)]
            audit.append(dict(url=url,parameters=p,file=name,row_start=y0,col_start=x0))
    image=Image.new('F',(cols,rows))
    image.putdata(values)
    buf=io.BytesIO()
    image.save(buf,format='TIFF')
    return values,buf.getvalue(),audit


def fetch_dem(site,output,fetch,api):
    # Reuse the public fetch_site constants, decoder and atomic writer.
    spec=grid_spec(site.get('bounds_ft'))
    bounds,rows,cols=spec['bounds_ft'],spec['rows'],spec['cols']
    bbox=bbox_mercator(site,bounds)
    place=jurisdiction(site,fetch)
    api.write_json(output/'jurisdiction.json',place)
    sources=[]
    if place['state']=='24':
        if place['county']=='021': sources.append(api.FREDERICK)
        sources.append(api.MARYLAND)
    sources.append(api.USGS)
    errors=[]
    for source in sources:
        try:
            if source==api.USGS and not (24<=site['lat']<=50 and -125<=site['lon']<=-66):
                raise ValueError('NAVD88 not guaranteed outside CONUS; need a datum-aware source')
            metadata=fetch.get_json(source+'?f=pjson')
            if metadata.get('bandCount')!=1: raise ValueError('Expected a single-band elevation service')
            tiled=False
            try:
                meters,raw,request=export(source,bbox,cols,rows,fetch,api.decode_dem,source==api.USGS)
                requests=[request]
            except Exception as exc:
                fetch.note(f'Full DEM export failed; trying four pixel-aligned tiles: {exc}')
                meters,raw,requests=export_tiled(source,bbox,cols,rows,fetch,api.decode_dem,output)
                tiled=True
            (output/'dem-source.tif').write_bytes(raw)
            api.write_json(output/'dem-service.json',metadata)
            result=dict(schema_version=1,**spec,units='feet',vertical_datum='NAVD88',horizontal_crs='EPSG:3857',
                mercator_center=site['mercator_center'],bbox_3857=bbox,row_order='north_to_south',
                column_order='west_to_east',sample_location='cell_center',axes=dict(x='east',z='south'),
                elevations_ft=[v/.3048 for v in meters],downloaded_utc=api.utc_now(),
                source=dict(url=source,service_url=source,export_url=requests[0]['url'],
                            parameters=parameters(bbox,cols,rows),requests=requests,tiled=tiled,
                            elevation_units='meters',vertical_datum='NAVD88',meters_per_output_foot=.3048))
            api.write_json(output/'dem.json',result)
            fetch.note(f'DEM saved: {rows}x{cols} cells, source={source}, tiled={tiled}')
            return result
        except Exception as exc:
            errors.append(f'{source}: {exc}')
            fetch.note(f'DEM source failed: {errors[-1]}')
    raise ValueError('All DEM sources failed: '+'; '.join(errors))
