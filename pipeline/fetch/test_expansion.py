import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fetch_site as fetch


class ExpansionTests(unittest.TestCase):
    def setup_site(self,root):
        folder=Path(root)/'test-site'
        folder.mkdir()
        old=dict(address='Test Site',slug='test-site',status='complete',lat=39.,lon=-77.,bounds_ft=[-100,100,-100,100])
        (folder/'site.json').write_text(json.dumps(old))
        (folder/'dem.json').write_text('old-dem')
        (folder/'index.html').write_text('existing-model')
        (folder/'features.json').write_text(json.dumps(dict(coverage=dict(expansions=0))))
        args=SimpleNamespace(expand_into=folder,bounds_ft=[-200,200,-200,200],address='Test Site',
                             lat=39.,lon=-77.,user_agent='test',osm_raw_json=None)
        return folder,args,old

    def test_failed_refetch_preserves_complete_site_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,args,old=self.setup_site(tmp)
            before=(folder/'site.json').read_bytes()
            with patch.object(fetch,'main',return_value=1):
                self.assertEqual(fetch.expand_existing(args),1)
            self.assertEqual((folder/'site.json').read_bytes(),before)
            self.assertEqual((folder/'dem.json').read_text(),'old-dem')
            self.assertFalse((folder/'.fetch-expand.lock').exists())

    def test_success_preserves_history_and_non_fetch_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,args,old=self.setup_site(tmp)
            def stage(tokens):
                out=Path(tokens[tokens.index('--out-root')+1])/'test-site'
                out.mkdir(parents=True)
                (out/'site.json').write_text(json.dumps(dict(old,bounds_ft=[-201,201,-201,201])))
                (out/'dem.json').write_text('new-dem')
                return 0
            with patch.object(fetch,'main',side_effect=stage):
                self.assertEqual(fetch.expand_existing(args),0)
            self.assertEqual((folder/'dem.json').read_text(),'new-dem')
            self.assertEqual((folder/'index.html').read_text(),'existing-model')
            saved=list((folder/'fetch-history').glob('*/dem.json'))
            self.assertEqual(len(saved),1)
            self.assertEqual(saved[0].read_text(),'old-dem')

    def test_changed_origin_and_second_expansion_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,args,old=self.setup_site(tmp)
            args.lat=40.
            with self.assertRaisesRegex(ValueError,'origin'): fetch.expand_existing(args)
            args.lat=39.
            (folder/'features.json').write_text(json.dumps(dict(coverage=dict(expansions=1))))
            with self.assertRaisesRegex(ValueError,'already attempted'): fetch.expand_existing(args)

    def test_promotion_error_restores_original_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,args,old=self.setup_site(tmp)
            def stage(tokens):
                out=Path(tokens[tokens.index('--out-root')+1])/'test-site'
                out.mkdir(parents=True)
                (out/'dem.json').write_text('new-dem')
                (out/'site.json').write_text(json.dumps(dict(old,bounds_ft=[-201,201,-201,201])))
                return 0
            original=Path.replace
            failed=[False]
            def replace(path,target):
                if path.name=='site.json' and path.parent.parent.name.startswith('.fetch-expansion') and not failed[0]:
                    failed[0]=True
                    raise OSError('simulated promotion failure')
                return original(path,target)
            with patch.object(fetch,'main',side_effect=stage),patch.object(Path,'replace',new=replace):
                with self.assertRaisesRegex(OSError,'promotion failure'): fetch.expand_existing(args)
            self.assertEqual(json.loads((folder/'site.json').read_text()),old)
            self.assertEqual((folder/'dem.json').read_text(),'old-dem')
            self.assertFalse((folder/'.fetch-expand.lock').exists())

    def test_comma_bounds_with_negative_first_value(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(fetch,'fetch_dem'), patch.object(fetch,'run_lidar'):
            result=fetch.main(['Test Site','--lat','39','--lon','-77','--no-features',
                               '--bounds-ft','-100,200,-50,100','--out-root',tmp])
            self.assertEqual(result,0)
            site=json.loads((Path(tmp)/'test-site/site.json').read_text())
            self.assertLessEqual(site['bounds_ft'][0],-100)
            self.assertGreaterEqual(site['bounds_ft'][1],200)


if __name__=='__main__': unittest.main()
