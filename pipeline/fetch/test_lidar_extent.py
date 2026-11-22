import tempfile
from pathlib import Path
import unittest
import json
import lidar_extent


class ExtentTests(unittest.TestCase):
    def test_dataset_edge_intersection_is_selected_without_center(self):
        g=dict(type='Polygon',coordinates=[[[9,0],[11,0],[11,2],[9,2],[9,0]]])
        self.assertTrue(lidar_extent.intersects(g,[0,0,10,10]))
        self.assertFalse(lidar_extent.intersects(g,[-10,-10,-1,-1]))

    def test_holes_do_not_count_as_coverage(self):
        g=dict(type='Polygon',coordinates=[[[0,0],[10,0],[10,10],[0,10],[0,0]],
                                           [[2,2],[8,2],[8,8],[2,8],[2,2]]])
        self.assertFalse(lidar_extent.intersects(g,[3,3,7,7]))

    def test_wrong_crs_and_non_las_data_are_rejected(self):
        class API:
            def get(self,url): return json.dumps(dict(srs={'horizontal':'26918'},bounds=[0,0,0,1,1,1])).encode()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'EPSG:3857'):
                lidar_extent.fetch_dataset(API(),'https://example.test/ept.json',[0,0,1,1],Path(temp))


if __name__=='__main__': unittest.main()
