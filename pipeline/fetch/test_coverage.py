import unittest
from coverage import contains, polygon_distance, plan, grid_spec, CELL_FT


def rectangle(name,x0,x1,z0,z1,holes=None):
    return dict(id=name,polygon=[[x0,z0],[x1,z0],[x1,z1],[x0,z1],[x0,z0]],holes=holes or [])


class CoverageTests(unittest.TestCase):
    def test_original_buffer_does_not_chain_neighbors(self):
        a=rectangle('target',-20,20,-20,20)
        b=rectangle('near',115,145,-10,10)
        c=rectangle('far',230,260,-10,10)
        spec,cov=plan(dict(status='ok',buildings=[a,b,c]))
        self.assertEqual(cov['selected_building_ids'],['target','near'])
        self.assertEqual(cov['target_building_id'],'target')
        self.assertGreaterEqual(spec['bounds_ft'][1],175)
        self.assertEqual(cov['status'],'complete')

    def test_crossing_edges_without_contained_vertices(self):
        a=rectangle('a',-100,100,-2,2)
        b=rectangle('b',-2,2,-100,100)
        self.assertEqual(polygon_distance(a,b),0)

    def test_courtyard_is_not_building_interior(self):
        a=rectangle('ring',-100,100,-100,100,[[[-80,-80],[80,-80],[80,80],[-80,80],[-80,-80]]])
        b=rectangle('courtyard',-5,5,-5,5)
        self.assertFalse(contains([0,0],a))
        self.assertEqual(polygon_distance(a,b),75)

    def test_diagonal_bbox_overlap_is_not_buffer_contact(self):
        a=rectangle('a',-10,10,-10,10)
        b=rectangle('b',100,110,100,110)
        _,cov=plan(dict(status='ok',buildings=[a,b]))
        self.assertEqual(cov['selected_building_ids'],['a'])

    def test_budget_clipping_and_intentional_sections_differ(self):
        a=rectangle('a',-800,800,-50,50)
        spec,cov=plan(dict(status='ok',buildings=[a]),expansions=1)
        self.assertLessEqual(spec['bounds_ft'][1]-spec['bounds_ft'][0],1500)
        self.assertEqual(cov['status'],'incomplete')
        self.assertIn('a',cov['edge_touching'])
        a=rectangle('a',-10,10,-10,10)
        b=rectangle('large-neighbor',100,500,-100,100)
        _,cov=plan(dict(status='ok',buildings=[a,b]))
        self.assertEqual(cov['section_cut_ids'],['large-neighbor'])
        self.assertEqual(cov['edge_touching'],[])

    def test_unmapped_target_stays_incomplete(self):
        _,cov=plan(dict(status='unavailable',buildings=[]),expansions=1)
        self.assertEqual(cov['status'],'incomplete')

    def test_unresolved_relation_inside_target_buffer_prevents_complete(self):
        f=dict(status='ok',buildings=[rectangle('a',-10,10,-10,10)],
               unresolved_geometries=[dict(id='relation/missing',bounds_ft=[80,90,0,10])])
        _,cov=plan(f)
        self.assertEqual(cov['status'],'incomplete')
        f['unresolved_geometries'][0]['bounds_ft']=[800,900,0,10]
        _,cov=plan(f)
        self.assertEqual(cov['status'],'complete')

    def test_non_square_asymmetric_grid_cell_edges(self):
        spec=grid_spec([-10,110,-40,10])
        a,b,c,d=spec['bounds_ft']
        self.assertLessEqual(a,-10)
        self.assertGreaterEqual(b,110)
        self.assertNotEqual(spec['rows'],spec['cols'])
        self.assertAlmostEqual(b-a,spec['cols']*CELL_FT)
        self.assertAlmostEqual(d-c,spec['rows']*CELL_FT)
        self.assertEqual(grid_spec()['rows'],120)
        self.assertEqual(grid_spec(spec['bounds_ft']),spec)
        self.assertGreaterEqual(grid_spec([0,1,0,1])['rows'],2)
        self.assertGreaterEqual(grid_spec([0,1,0,1])['cols'],2)

    def test_oversized_asymmetric_target_keeps_address_in_data(self):
        a=rectangle('target',-10,3000,-10,50)
        spec,cov=plan(dict(status='ok',buildings=[a]))
        b=spec['bounds_ft']
        self.assertLess(b[0],0)
        self.assertGreater(b[1],0)
        self.assertEqual(cov['status'],'incomplete')

    def test_expansion_status_and_invalid_bounds(self):
        _,cov=plan(dict(status='ok',buildings=[rectangle('a',-10,10,-10,10)]),expansions=1)
        self.assertEqual(cov['status'],'expanded')
        for b in ([0,0,0,1],[0,float('nan'),0,1],[1,0,0,1]):
            with self.assertRaises(ValueError): grid_spec(b)


if __name__=='__main__': unittest.main()
