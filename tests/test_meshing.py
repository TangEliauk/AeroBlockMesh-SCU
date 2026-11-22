"""Structured O-grid topology and PLOT3D round trip."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from smartcfd.core.io import write_plot3d
from smartcfd.geometry.airfoil import naca
from smartcfd.meshing.structured import BASE_PARAMS, generate


class StructuredMeshTests(unittest.TestCase):
    def test_farfield_and_matching_seam(self):
        case = naca("2412")
        parameters = [0.6, 1.0, 0.2, -0.3, 0.2]
        for fine, rows, columns in ((False, 17, 97), (True, 33, 161)):
            with self.subTest(fine=fine):
                mesh = generate(case, parameters, fine=fine)
                self.assertEqual(mesh.shape, (rows, columns, 2))
                np.testing.assert_array_equal(mesh[:, 0], mesh[:, -1])
                np.testing.assert_array_equal(mesh[0, 0], [1, 0])
                np.testing.assert_array_equal(mesh[0, (columns - 1) // 2], [0, 0])
                farfield_radii = np.linalg.norm(mesh[-1] - [0.5, 0], axis=1)
                np.testing.assert_allclose(farfield_radii, 4.0)

    def test_invalid_parameters(self):
        case = naca("2412")
        for parameters in ([], [0, 0, np.nan, 0, 0]):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                generate(case, parameters)

    def test_plot3d_preserves_all_nodes_and_interface_order(self):
        mesh = generate(naca("2412"), BASE_PARAMS)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "naca2412.x"
            ni, nj, nk = write_plot3d(path, mesh)
            self.assertEqual((ni, nj, nk), (25, 17, 1))
            with path.open(encoding="ascii") as stream:
                self.assertEqual(int(stream.readline()), 4)
                for _ in range(4):
                    self.assertEqual(tuple(map(int, stream.readline().split())), (ni, nj, nk))
                blocks = np.loadtxt(stream).reshape(4, 3, nj, ni).transpose(0, 2, 3, 1)

        np.testing.assert_array_equal(blocks[..., 2], 0.0)
        for index in range(4):
            np.testing.assert_array_equal(blocks[index, :, -1], blocks[(index + 1) % 4, :, 0])
            first = index * (ni - 1)
            np.testing.assert_array_equal(blocks[index, :, :, :2], mesh[::-1, first : first + ni])

        along_i = blocks[:, :-1, 1:, :2] - blocks[:, :-1, :-1, :2]
        along_j = blocks[:, 1:, :-1, :2] - blocks[:, :-1, :-1, :2]
        signed = along_i[..., 0] * along_j[..., 1] - along_i[..., 1] * along_j[..., 0]
        self.assertTrue(np.all(signed > 0.0))
