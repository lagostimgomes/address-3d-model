import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse
import urllib.error

from PIL import Image
import fetch_site as api
import dem_grid


def tiff(cols,rows,values):
    im=Image.new('F',(cols,rows)); im.putdata(values)
    buf=io.BytesIO(); im.save(buf,format='TIFF')
    return buf.getvalue()


class DEMTests(unittest.TestCase):
    def test_jurisdiction_excludes_dc_and_unknown(self):
        self.assertEqual(api.dem_sources(38.9,-77.04,state='11'),[api.USGS])
        self.assertEqual(api.dem_sources(39.4,-77.4),[api.USGS])
        self.assertEqual(api.dem_sources(39.4,-77.4,state='24',county='021'),[api.FREDERICK,api.MARYLAND,api.USGS])

    def test_geography_lookup_failure_uses_national_only(self):
        class Client:
            def get_json(self,url): raise OSError('offline')
            def note(self,msg): pass
        self.assertIsNone(dem_grid.jurisdiction(dict(lat=39,lon=-77),Client())['state'])

    def test_tiled_stitch_has_no_flip_shift_or_odd_size_seam(self):
        # 5x3 deliberately splits at unequal pixel counts.
        class Client:
            def get(self,url,**kw):
                p=urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
                cols,rows=map(int,p['size'][0].split(','))
                x0,y0,x1,y1=map(float,p['bbox'][0].split(','))
                startrow=round(30-y1); startcol=round(x0-10)
                return tiff(cols,rows,[100*(startrow+r)+startcol+c for r in range(rows) for c in range(cols)])
        with tempfile.TemporaryDirectory() as temp:
            vals,raw,audit=dem_grid.export_tiled('https://example.test',[10,27,15,30],5,3,Client(),api.decode_dem,Path(temp))
        self.assertEqual(vals,[100*r+c for r in range(3) for c in range(5)])
        self.assertEqual(api.decode_dem(raw,cols=5,rows=3),vals)
        self.assertEqual(len(audit),4)

    def test_retry_schedule_and_non_retryable_http(self):
        errors=[urllib.error.HTTPError('u',502,'bad',{},None)]*3
        class Response:
            status=200; url='u'
            def read(self): return b'ok'
            def __enter__(self): return self
            def __exit__(self,*args): pass
        fetch=api.Fetcher(io.StringIO(),'test')
        with patch.object(api.urllib.request,'urlopen',side_effect=errors+[Response()]), patch.object(api.time,'sleep') as sleep:
            self.assertEqual(fetch.get('https://example.test',attempts=4,backoff=(5,15,30)),b'ok')
            self.assertEqual([c.args[0] for c in sleep.call_args_list],[5,15,30])
        with patch.object(api.urllib.request,'urlopen',side_effect=urllib.error.HTTPError('u',400,'bad',{},None)) as request:
            with self.assertRaises(urllib.error.HTTPError): fetch.get('https://example.test',attempts=4)
            self.assertEqual(request.call_count,1)

    def test_full_export_failure_falls_back_and_records_provenance(self):
        calls=[]
        class Client:
            def get_json(self,url):
                if '/geographies/' in url:
                    return {'result':{'geographies':{'States':[{'STATE':'11'}]}}}
                calls.append(url)
                return dict(bandCount=1)
            def get(self,url,**kwargs):
                p=urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
                cols,rows=map(int,p['size'][0].split(','))
                if cols==120 and rows==120: raise OSError('full request rejected')
                return tiff(cols,rows,[10.]*(cols*rows))
            def note(self,msg): pass
        site=dict(lat=38.9,lon=-77.04,mercator_center=api.mercator(38.9,-77.04))
        with tempfile.TemporaryDirectory() as temp:
            result=dem_grid.fetch_dem(site,Path(temp),Client(),api)
            self.assertTrue((Path(temp)/'dem.json').exists())
            self.assertTrue(result['source']['tiled'])
            self.assertEqual(len(result['elevations_ft']),14400)
            self.assertAlmostEqual(result['elevations_ft'][0],10/.3048)
        self.assertEqual(calls,[api.USGS+'?f=pjson'])

    def test_missing_tile_fails_without_publishing_dem(self):
        class Client:
            def get_json(self,url): return dict(bandCount=1)
            def get(self,url,**kwargs): raise OSError('forced failure')
            def note(self,msg): pass
        site=dict(lat=38.9,lon=-77.04,mercator_center=api.mercator(38.9,-77.04))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'All DEM sources failed'):
                dem_grid.fetch_dem(site,Path(temp),Client(),api)
            self.assertFalse((Path(temp)/'dem.json').exists())


if __name__=='__main__': unittest.main()
