import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from smartcfd.pipeline.workflow import export_naca_mesh


class NacaPipelineTests(unittest.TestCase):
    def test_export_writes_mesh_and_interface_metadata_without_overwriting(self):
        with TemporaryDirectory() as directory:
            grid = Path(directory) / "naca_2412.x"
            report = export_naca_mesh("2412", grid, fine=True)
            raw = grid.read_bytes()
            metadata = json.loads(grid.with_suffix(".interfaces.json").read_text(encoding="utf-8"))
            self.assertTrue(raw.startswith(b"4\n"))
            self.assertEqual(metadata, report)
            self.assertEqual(report["metrics"]["cells"], 5120)
            self.assertEqual(len(report["interfaces"]), 4)
            self.assertEqual(report["metrics"]["invalid_cells"], 0)
            with self.assertRaises(FileExistsError):
                export_naca_mesh("2412", grid, fine=True)
            self.assertEqual(grid.read_bytes(), raw)

    def test_invalid_fixed_grid_does_not_create_files(self):
        with TemporaryDirectory() as directory:
            grid = Path(directory) / "naca_6409.x"
            with self.assertRaisesRegex(ValueError, "1 invalid"):
                export_naca_mesh("6409", grid, method="fixed")
            self.assertEqual(list(Path(directory).iterdir()), [])
