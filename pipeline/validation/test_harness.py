import copy
import unittest
import harness as h

def row(i,lat=0,lon=0,**extra):
    return dict(id=str(i),site_id='site'+str(i),group_id='group'+str(i),group_basis='synthetic test',source='synthetic',lat=lat,lon=lon,**extra)

class HarnessTests(unittest.TestCase):
    def test_development_flag_excludes_entire_connected_group(self):
        rows=[row('a',development_only=True),row('b',lon=.001),row('c',lon=1)]
        m=h.manifest(rows,[])
        self.assertEqual(m['excluded_development_ids'],['a','b'])
        self.assertEqual(m['selected_groups'],1)

    def test_neighbours_repeat_surveys_and_development_excluded_before_split(self):
        rows=[row('a'),row('b',lon=.001),row('c',lon=1),row('d',lon=2)]
        rows[1]['site_id']=rows[0]['site_id']
        m=h.manifest(rows,[row('dev',lon=2.001)],target_groups=60)
        self.assertIn('d',m['excluded_development_ids'])
        group=next(g for g in m['groups'] if any(r['id']=='a' for r in g['records']))
        self.assertEqual({r['id'] for r in group['records']},{'a','b'})
        self.assertEqual(m,h.manifest(list(reversed(rows)),[row('dev',lon=2.001)],target_groups=60))

    def test_geographic_groups_never_cross_split_and_missing_labels_explicit(self):
        rows=[row(i,lon=i*.03) for i in range(12)]
        m=h.manifest(rows,[],target_groups=12)
        self.assertEqual(m['group_counts'],{'tuning':4,'calibration':4,'evaluation':4})
        self.assertEqual(m['groups'][0]['strata'],[['missing','missing','missing']])
        for a in m['groups']:
            for b in m['groups']:
                if a['split']!=b['split']:
                    for x in a['records']:
                        for y in b['records']: self.assertGreater(h.distance(x,y),1000)

    def test_missing_truth_and_wrong_split_never_become_pass(self):
        m=h.manifest([row(i,lon=i*.03) for i in range(6)],[])
        s=h.summarize(m,[],'evaluation')
        self.assertIsNone(s['metrics']['material_correct']['mean'])
        self.assertEqual(s['metrics']['material_correct']['status'],'incomplete')
        rid=next(g['records'][0]['id'] for g in m['groups'] if g['split']=='tuning')
        with self.assertRaises(ValueError):h.summarize(m,[{'id':rid}],'evaluation')
        bad=copy.deepcopy(m);bad['seed']='changed'
        with self.assertRaises(ValueError):h.summarize(bad,[],'evaluation')

    def test_invalid_coordinates_duplicate_ids_and_unattributed_strata_rejected(self):
        for rows in ([row(1),row(1)], [row(1,lat=float('nan'))], [row(1,strata={'use':'school'})]):
            with self.assertRaises(ValueError):h.manifest(rows,[])

if __name__=='__main__':unittest.main()
