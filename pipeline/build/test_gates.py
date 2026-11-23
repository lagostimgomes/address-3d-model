"""Regression tests for gates.py, including Codex's R2 adversarial fixtures (review-r2/audit.py).
Run: python -m unittest discover -s pipeline/build -p 'test_*.py'"""
import base64, json, os, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates


def packed(tris):
    a = np.round(np.asarray(tris, float) * 100).astype('<i4')
    return {'p': base64.b64encode(a.tobytes()).decode(), 'n': int(a.size), 'q': 100}


V = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]], float)
TET = V[np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])]          # outward-oriented closed tetrahedron


def run_case(houses, heldout=None):
    d = tempfile.mkdtemp()
    json.dump({'houses': houses}, open(os.path.join(d, 'scene.json'), 'w'))
    if heldout is not None: json.dump(heldout, open(os.path.join(d, 'heldout.json'), 'w'))
    return gates.run(d)


def house(roof=None, walls=None, base=None, parts=None, hid='fixture'):
    m = {'base': base or []}
    if roof is not None: m['roof'] = packed(roof)
    if walls is not None: m['walls'] = packed(walls)
    return {'id': hid, 'type': 'mesh', 'mesh': m, 'parts': parts or []}


def box(x0, z0, x1, z1, y0, y1):
    """Closed, outward-oriented box: walls in 'walls', top in 'roof', bottom left open as a declared base."""
    c = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    walls, base = [], []
    for (ax, az), (bx, bz) in zip(c, c[1:] + c[:1]):
        A, B, C, D = (ax, y0, az), (bx, y0, bz), (bx, y1, bz), (ax, y1, az)
        # outward for a counter-clockwise footprint seen from above in (x, z) with y up: (A, D, C), (A, C, B)
        walls += [(A, D, C), (A, C, B)]
        base.append([list(A), list(B)])
    top = [((x0, y1, z0), (x0, y1, z1), (x1, y1, z1)), ((x0, y1, z0), (x1, y1, z1), (x1, y1, z0))]
    return np.array(top, float), np.array(walls, float), base


class AdversarialR2(unittest.TestCase):
    def test_reference_closed_shell_passes_geometry(self):
        r = run_case([house(roof=TET)])
        b = r['buildings'][0]
        self.assertEqual(b['topology']['verdict'], 'pass')
        self.assertEqual(b['orientation']['verdict'], 'pass')
        self.assertEqual(b['intersections']['verdict'], 'pass')

    def test_interpenetrating_closed_shells_fail(self):
        r = run_case([house(roof=np.concatenate([TET, TET + 1]))])
        self.assertEqual(r['buildings'][0]['intersections']['verdict'], 'fail')
        self.assertEqual(r['acceptance'], 'fail')

    def test_fully_inverted_shell_fails(self):
        r = run_case([house(roof=TET[:, ::-1])])
        self.assertEqual(r['buildings'][0]['orientation']['verdict'], 'fail')
        self.assertEqual(r['acceptance'], 'fail')

    def test_arbitrary_missing_face_declared_base_fails(self):
        base = [[V[0].tolist(), V[2].tolist()], [V[2].tolist(), V[1].tolist()], [V[1].tolist(), V[0].tolist()]]
        r = run_case([house(roof=TET[1:], base=base)])
        self.assertEqual(r['buildings'][0]['base']['verdict'], 'fail')
        self.assertEqual(r['acceptance'], 'fail')

    def test_failed_heldout_fails(self):
        top, walls, base = box(0, 0, 40, 40, 0, 20)
        pts = [[x, 25.0, z, (int(x // 10) * 100003 + int(z // 10))] for x in np.linspace(1, 39, 10) for z in np.linspace(1, 39, 10)]
        r = run_case([house(roof=top, walls=walls, base=base, parts=[{'id': 'p0', 'kind': 'flat'}])], heldout={'fixture': {'p0': pts}})
        b = r['buildings'][0]
        self.assertEqual(b['heldout']['verdict'], 'fail')
        self.assertEqual(r['acceptance'], 'fail')

    def test_missing_heldout_is_incomplete_not_pass(self):
        top, walls, base = box(0, 0, 40, 40, 0, 20)
        r = run_case([house(roof=top, walls=walls, base=base, parts=[{'id': 'p0', 'kind': 'envelope'}])], heldout={'fixture': {}})
        self.assertEqual(r['buildings'][0]['heldout']['verdict'], 'incomplete')
        self.assertNotEqual(r['acceptance'], 'pass')

    def test_good_heldout_and_box_pass_except_terrain(self):
        top, walls, base = box(0, 0, 40, 40, 0, 20)
        pts = [[x, 20.1, z, (int(x // 10) * 100003 + int(z // 10))] for x in np.linspace(1, 39, 10) for z in np.linspace(1, 39, 10)]
        r = run_case([house(roof=top, walls=walls, base=base, parts=[{'id': 'p0', 'kind': 'flat'}])], heldout={'fixture': {'p0': pts}})
        b = r['buildings'][0]
        self.assertEqual(b['topology']['verdict'], 'pass')
        self.assertEqual(b['orientation']['verdict'], 'pass')
        self.assertEqual(b['heldout']['verdict'], 'pass')
        self.assertEqual(b['base']['verdict'], 'incomplete')     # no dem.json: base height cannot be verified
        self.assertEqual(r['acceptance'], 'incomplete')

    def test_empty_population_never_passes(self):
        r = run_case([{'id': 'h', 'type': 'house'}])
        self.assertEqual(r['acceptance'], 'incomplete')            # unchecked building keeps the site incomplete
        r = run_case([])
        self.assertEqual(r['acceptance'], 'not_applicable')


if __name__ == '__main__':
    unittest.main()
