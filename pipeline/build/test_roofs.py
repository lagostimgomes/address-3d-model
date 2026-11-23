"""Plausibility bounds on roof candidates (provisional; review s2-bound)."""
import unittest
import numpy as np
from shapely.geometry import Point, box
import roofs as RF


def dome_pts(H, R=20, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-R, R, 4000); z = rng.uniform(-R, R, 4000); r = np.hypot(x, z); m = r < R * 0.975
    x, z, r = x[m], z[m], r[m]
    return np.c_[x, 50 + H * np.sqrt(1 - (r / R) ** 2) + rng.normal(0, 0.1, len(r)), z]


class Bounds(unittest.TestCase):
    def test_dome_within_bound_selected(self):
        f, st = RF.select(dome_pts(25), Point(0, 0).buffer(20, quad_segs=32))
        self.assertEqual(st['kind'], 'dome'); self.assertAlmostEqual(st['shape']['rise_over_radius'], 1.25, places=1)

    def test_dome_over_bound_rejected_and_recorded(self):
        f, st = RF.select(dome_pts(40), Point(0, 0).buffer(20, quad_segs=32))
        self.assertNotEqual(st['kind'], 'dome')
        self.assertTrue(any('dome rise/radius' in a.get('rejected', '') for a in st['alternatives']))

    def test_steep_plane_rejected(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 10, 2000); z = rng.uniform(0, 10, 2000)
        f, st = RF.select(np.c_[x, 10 + 3.0 * x + rng.normal(0, 0.05, len(x)), z], box(0, 0, 10, 10))
        self.assertNotIn(st['kind'], ('shed', 'gable', 'hip'))

    def test_moderate_plane_kept(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 30, 3000); z = rng.uniform(0, 20, 3000)
        f, st = RF.select(np.c_[x, 10 + 0.5 * x + rng.normal(0, 0.05, len(x)), z], box(0, 0, 30, 20))
        self.assertEqual(st['kind'], 'shed')


if __name__ == '__main__':
    unittest.main()
