"""Geometric score and signed cell-quality checks."""

import unittest

import numpy as np

from smartcfd.evaluation.mesh_quality import cell_quality, quality
from smartcfd.geometry.airfoil import naca
from smartcfd.meshing.structured import BASE_PARAMS, generate


class MeshQualityTests(unittest.TestCase):
    def test_square_quality_and_inversion(self):
        square = np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
        scores = quality(square)
        self.assertEqual(scores["invalid_cells"], 0)
        self.assertAlmostEqual(scores["score"], 1.0)
        self.assertAlmostEqual(scores["mean_skew"], 0.0)
        self.assertAlmostEqual(scores["max_aspect_ratio"], 1.0)
        self.assertEqual(quality(square[:, ::-1])["invalid_cells"], 1)
        self.assertAlmostEqual(cell_quality(square[:, ::-1]).item(), -1.0)

    def test_quality_field_is_signed_and_scale_invariant(self):
        baseline = generate(naca("6409"), BASE_PARAMS)
        jacobian = cell_quality(baseline)
        metrics = quality(baseline)
        self.assertEqual(jacobian.shape, (16, 96))
        self.assertEqual(metrics["cells"], jacobian.size)
        self.assertEqual(metrics["invalid_cells"], 1)
        self.assertAlmostEqual(metrics["min_jacobian"], jacobian.min())
        np.testing.assert_allclose(cell_quality(3.5 * baseline + [1.2, -0.5]), jacobian, atol=1e-12)
