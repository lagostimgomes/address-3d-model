import json
import math
from pathlib import Path
import tempfile
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

import fetch_osm as osm

SITE=dict(lat=0.,lon=0.,mercator_center=[0.,0.])


def geo(points):
    return [dict(lon=math.degrees(x*.3048/osm.R),lat=math.degrees(2*math.atan(math.exp(-z*.3048/osm.R))-math.pi/2)) for x,z in points]


def way(i,points,tags):
    return dict(type='way',id=i,geometry=geo(points),tags=tags,timestamp='2026-01-01T00:00:00Z')


class OSMTests(unittest.TestCase):
    def test_post_failover_and_transient_retry(self):
        err=urllib.error.HTTPError('https://example.test',503,'busy',{},None)
        with patch.object(osm,'ENDPOINTS',['https://a.test','https://b.test']), patch.object(osm.time,'sleep') as sleep, \
             patch.object(osm.urllib.request,'urlopen',side_effect=[err,err,BytesIO(b'{"elements":[]}')]) as call:
            data,url=osm.overpass('query',[])
        self.assertEqual(url,'https://a.test')
        self.assertEqual(data,{'elements':[]})
        self.assertEqual(call.call_args_list[0].args[0].get_method(),'POST')
        self.assertEqual(call.call_args_list[0].args[0].data,b'data=query')
        sleep.assert_called_once_with(2.0)

    def test_permanent_http_and_partial_results_not_retried(self):
        err=urllib.error.HTTPError('https://example.test',400,'bad query',{},None)
        with patch.object(osm,'ENDPOINTS',['https://a.test','https://b.test']), patch.object(osm.time,'sleep') as sleep, \
             patch.object(osm.urllib.request,'urlopen',side_effect=[err,BytesIO(b'{"elements":[],"remark":"timeout"}')]) as call:
            with self.assertRaises(RuntimeError): osm.overpass('query',[])
        self.assertEqual(call.call_count,2); sleep.assert_not_called()

    def test_rate_limit_cooldown_outside_budget_not_retried(self):
        err=urllib.error.HTTPError('https://example.test',429,'busy',{'Retry-After':'120'},None)
        with patch.object(osm,'ENDPOINTS',['https://a.test']), patch.object(osm.time,'sleep') as sleep, \
             patch.object(osm.urllib.request,'urlopen',side_effect=err) as call:
            with self.assertRaises(RuntimeError): osm.overpass('query',[])
        self.assertEqual(call.call_count,1); sleep.assert_not_called()

    def test_validated_raw_cache_preserves_empty_result_as_stale(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            with patch.object(osm,'overpass',return_value=({'elements':[]},'https://a.test')):
                fresh=osm.fetch_features(SITE,folder)
            acquired=fresh['acquisition']['acquired_utc']
            (folder/'features.json').unlink()
            with patch.object(osm,'overpass',side_effect=OSError('offline')):
                stale=osm.fetch_features(SITE,folder)
            self.assertEqual(stale['status'],'stale')
            self.assertEqual(stale['endpoint'],'validated-raw-cache')
            self.assertEqual(stale['acquisition']['acquired_utc'],acquired)
            self.assertEqual(stale['buildings'],[])

    def test_raw_cache_rejects_different_frame_and_tampered_response(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            with patch.object(osm,'overpass',return_value=({'elements':[]},'https://a.test')):
                osm.fetch_features(SITE,folder)
            with patch.object(osm,'overpass',side_effect=OSError('offline')):
                other=osm.fetch_features(dict(SITE,mercator_center=[1,1]),folder)
            self.assertEqual(other['status'],'unavailable')
            (folder/'osm-raw.json').write_text('{"elements":[],"changed":true}')
            with patch.object(osm,'overpass',side_effect=OSError('offline')):
                invalid=osm.fetch_features(SITE,folder)
            self.assertEqual(invalid['status'],'unavailable')

    def test_multipolygon_reversed_members_holes_and_no_duplicate_members(self):
        a=[[-20,-20],[20,-20],[20,20]]
        b=[[-20,-20],[-20,20],[20,20]]
        h=[[-5,-5],[5,-5],[5,5],[-5,5],[-5,-5]]
        rel=dict(type='relation',id=10,tags={'type':'multipolygon','building':'school'},members=[
            dict(type='way',ref=1,role='outer',geometry=geo(a)),
            dict(type='way',ref=2,role='outer',geometry=geo(b)),
            dict(type='way',ref=3,role='inner',geometry=geo(h))])
        result=osm.normalize(dict(elements=[rel,way(3,h,{'building':'yes'})]),SITE)
        self.assertEqual(len(result['buildings']),1)
        building=result['buildings'][0]
        self.assertEqual(len(building['holes']),1)
        self.assertIsNone(building['source_date'])
        self.assertFalse(osm.contains([0,0],building))
        self.assertEqual(building['source_id'],'relation/10')

    def test_incomplete_relation_is_skipped_and_known_way_retained(self):
        ring=[[-20,-20],[20,-20],[20,20],[-20,-20]]
        rel=dict(type='relation',id=10,tags={'type':'multipolygon','building':'yes'},members=[
            dict(type='way',ref=1,role='outer',geometry=geo(ring)),
            dict(type='way',ref=2,role='outer')])
        result=osm.normalize(dict(elements=[rel,way(1,ring,{'building':'yes'})]),SITE)
        self.assertEqual(len(result['buildings']),1)
        self.assertEqual(result['buildings'][0]['source_id'],'way/1')
        self.assertIn('incomplete',result['notes'][0])

    def test_building_on_landuse_hole_retains_its_own_semantics(self):
        outer=[[-100,-100],[100,-100],[100,100],[-100,100],[-100,-100]]
        inner=[[-20,-20],[20,-20],[20,20],[-20,20],[-20,-20]]
        rel=dict(type='relation',id=10,tags={'type':'multipolygon','landuse':'grass'},members=[
            dict(type='way',ref=1,role='outer',geometry=geo(outer)),
            dict(type='way',ref=2,role='inner',geometry=geo(inner))])
        result=osm.normalize(dict(elements=[rel,way(2,inner,{'building':'yes'})]),SITE)
        self.assertEqual(len(result['buildings']),1)
        self.assertEqual(len(result['surfaces']),1)
        self.assertEqual(len(result['surfaces'][0]['holes']),1)

    def test_failed_refresh_retains_mapped_evidence_and_marks_stale(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'features.json').write_text(json.dumps(dict(status='ok',buildings=[dict(id='old',source_id='way/1',polygon=[[0,0],[1,0],[1,1],[0,0]],holes=[],tags={})],
                surfaces=[],ways=[],entrances=[],coverage=dict(status='incomplete'))))
            with patch.object(osm,'overpass',side_effect=OSError('offline')):
                f=osm.fetch_features(SITE,folder)
            self.assertEqual(f['status'],'stale')
            self.assertEqual(f['buildings'][0]['id'],'old')

    def test_surface_road_and_entrance_are_distinct_evidence(self):
        ring=[[-20,-20],[20,-20],[20,20],[-20,20],[-20,-20]]
        ent=dict(type='node',id=4,tags={'entrance':'main'},**geo([[20,0]])[0])
        result=osm.normalize(dict(elements=[way(1,ring,{'building':'school','building:levels':'2'}),
            way(2,[[20,0],[100,0]],{'highway':'service','width':'12 ft','lanes':'1'}),
            way(3,[[30,30],[80,30],[80,80],[30,30]],{'amenity':'parking'}),ent]),SITE)
        self.assertEqual(result['surfaces'][0]['kind'],'parking')
        self.assertEqual(result['ways'][0]['width_ft'],12)
        self.assertEqual(result['entrances'][0]['building_id'],'way/1')
        self.assertEqual(result['buildings'][0]['tags']['building:levels'],'2')

    def test_no_width_guess_and_unknown_landuse_is_not_lawn(self):
        self.assertIsNone(osm.width_feet('12;15'))
        self.assertIsNone(osm.width_feet('-3'))
        self.assertAlmostEqual(osm.width_feet('3.048 m'),10)
        self.assertEqual(osm.width_feet('10\'6"'),10.5)
        self.assertEqual(osm.surface_kind({'landuse':'residential'}),'other')
        self.assertEqual(osm.surface_kind({'landuse':'grass'}),'lawn')

    def test_unavailable_maps_are_explicit_and_legacy_files_still_exist(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(osm,'overpass',side_effect=OSError('offline')):
            result=osm.fetch_features(SITE,Path(temp))
            self.assertEqual(result['status'],'unavailable')
            self.assertEqual(result['surfaces'],[])
            self.assertTrue((Path(temp)/'features.json').exists())
            roads=json.loads((Path(temp)/'roads.json').read_text())
            self.assertEqual(roads['status'],'unavailable')
            self.assertEqual(roads['roads'],[])

    def test_query_contains_relations_nodes_tags_and_meta(self):
        q=osm.query_for(SITE,bounds_ft=[0,1500,-500,500])
        self.assertIn('relation["type"="multipolygon"]',q)
        self.assertIn('building|building:part',q)
        self.assertIn('node["entrance"]',q)
        self.assertIn('out meta geom',q)
        self.assertNotIn('/api/status',q)


if __name__=='__main__': unittest.main()
