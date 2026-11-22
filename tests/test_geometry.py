"""NACA input validation and contour geometry."""

import unittest

import numpy as np

from smartcfd.geometry.airfoil import airfoil_surface, case_features, naca


class AirfoilGeometryTests(unittest.TestCase):
    def test_naca_code_and_features(self):
        case = naca("2412")
        self.assertEqual((case.name, case.m, case.p, case.t), ("2412", 0.02, 0.4, 0.12))
        np.testing.assert_allclose(case_features(case), [1 / 3, 0.4, 0.5])

    def test_cusp_closure_and_symmetric_surfaces(self):
        angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        wall = airfoil_surface(naca("0012"), angles)
        np.testing.assert_array_equal(wall[[0, 2, 4]], [[1, 0], [0, 0], [1, 0]])
        np.testing.assert_allclose(wall[1], wall[3] * [1, -1])
        self.assertGreater(wall[1, 1], 0)

    def test_naca_rejects_unsupported_inputs(self):
        for code in ("240", "NACA2412", "241２", "0412", "2812", "7412", "2407", "2425"):
            with self.subTest(code=code), self.assertRaises(ValueError):
                naca(code)
