"""Exercise corrupt CGNS topology without editing or relying on real source cases."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/data/clean_cgns_multiblock.py"
SPEC = importlib.util.spec_from_file_location("clean_cgns_multiblock", SCRIPT)
CLEAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLEAN)


def node(parent, name, node_label, values=None):
    result = parent.create_group(name)
    result.attrs["label"] = np.bytes_(node_label)
    if isinstance(values, str):
        values = np.frombuffer(values.encode("ascii"), dtype="S1")
    if values is not None:
        result.create_dataset(" data", data=values)
    return result


def fixture(path):
    """Two adjacent 2x3x2 blocks, a six-vertex interface and exterior walls."""
    with h5py.File(path, "w") as handle:
        base = node(handle, "Base", "CGNSBase_t", np.array([3, 3], dtype="i4"))
        for name, x_offset in (("A", 0), ("B", 1)):
            zone = node(base, name, "Zone_t", np.array([[2, 3, 2], [1, 2, 1], [0, 0, 0]], dtype="i4"))
            node(zone, "ZoneType", "ZoneType_t", "Structured")
            coordinates = node(zone, "GridCoordinates", "GridCoordinates_t")
            k, j, i = np.indices((2, 3, 2), dtype=float)
            for axis, values in zip("XYZ", (i + x_offset, j, k)):
                node(coordinates, "Coordinate" + axis, "DataArray_t", values)
            container = node(zone, "Connectivity", "ZoneGridConnectivity_t")
            left = [[2, 1, 1], [2, 3, 2]]
            right = [[1, 1, 1], [1, 3, 2]]
            if name == "B":
                left, right = right, left
            interface = node(container, "to_" + ("B" if name == "A" else "A"),
                             "GridConnectivity1to1_t", "B" if name == "A" else "A")
            node(interface, "PointRange", "IndexRange_t", np.array(left, dtype="i4"))
            node(interface, "PointRangeDonor", "IndexRange_t", np.array(right, dtype="i4"))
            node(interface, "Transform", "int[IndexDimension]", np.array([1, 2, 3], dtype="i4"))
            boundary = node(node(zone, "ZoneBC", "ZoneBC_t"), "Wall", "BC_t", "BCWall")
            normal = 1 if name == "A" else 2
            node(boundary, "PointRange", "IndexRange_t",
                 np.array([[normal, 1, 1], [normal, 3, 2]], dtype="i4"))


class CGNSTopologyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "sample.h5"
        fixture(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def clean(self, source=None):
        return CLEAN.clean_case(source or self.source, self.root, "fixture")

    def test_extraction_is_deterministic_and_preserves_full_source(self):
        raw = self.source.read_bytes()
        compressed = self.root / "sample.cgns.gz"
        compressed.write_bytes(gzip.compress(raw, mtime=0))
        before = compressed.read_bytes()
        result = self.clean(compressed)
        self.assertEqual(CLEAN.encoded(result), CLEAN.encoded(self.clean(compressed)))
        self.assertEqual(compressed.read_bytes(), before)
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(result["checks"]["unique_interfaces"], 1)
        self.assertEqual(result["checks"]["matched_coordinate_pairs_including_both_directions"], 12)
        self.assertEqual(result["blocks"][0]["vertex_dimensions_ijk"], [2, 3, 2])
        self.assertEqual(len(result["boundaries"]), 2)
        self.assertEqual(result["checks"]["max_absolute_xyz_interface_difference"], [0, 0, 0])

    def test_non_permutation_transform_is_rejected(self):
        with h5py.File(self.source, "r+") as handle:
            handle["/Base/A/Connectivity/to_B/Transform/ data"][...] = [1, 1, 3]
        with self.assertRaisesRegex(CLEAN.TopologyError, "signed permutation"):
            self.clean()

    def test_interface_outside_zone_is_rejected_before_indexing(self):
        with h5py.File(self.source, "r+") as handle:
            handle["/Base/A/Connectivity/to_B/PointRange/ data"][...] = [[2, 1, 1], [2, 4, 2]]
        with self.assertRaisesRegex(CLEAN.TopologyError, "outside zone dimensions"):
            self.clean()

    def test_coordinate_mismatch_is_not_silently_repaired(self):
        with h5py.File(self.source, "r+") as handle:
            handle["/Base/B/GridCoordinates/CoordinateY/ data"][0, 0, 0] = 0.25
        raw = self.source.read_bytes()
        with self.assertRaisesRegex(CLEAN.TopologyError, "coordinate mismatch"):
            self.clean()
        self.assertEqual(self.source.read_bytes(), raw)

    def test_inverse_normal_transform_is_checked_even_on_a_fixed_face_axis(self):
        with h5py.File(self.source, "r+") as handle:
            handle["/Base/B/Connectivity/to_A/Transform/ data"][...] = [-1, 2, 3]
        with self.assertRaisesRegex(CLEAN.TopologyError, "not inverse"):
            self.clean()

    def test_missing_reverse_connectivity_is_rejected(self):
        with h5py.File(self.source, "r+") as handle:
            del handle["/Base/B/Connectivity/to_A"]
        with self.assertRaisesRegex(CLEAN.TopologyError, "two reciprocal"):
            self.clean()

    def test_reversed_ranges_and_signed_transform_map_the_same_geometry(self):
        with h5py.File(self.source, "r+") as handle:
            values = handle["/Base/B/GridCoordinates/CoordinateY/ data"]
            values[...] = 2 - values[()]
            forward = handle["/Base/A/Connectivity/to_B"]
            forward["PointRangeDonor/ data"][...] = [[1, 3, 1], [1, 1, 2]]
            forward["Transform/ data"][...] = [1, -2, 3]
            reverse = handle["/Base/B/Connectivity/to_A"]
            reverse["PointRangeDonor/ data"][...] = [[2, 3, 1], [2, 1, 2]]
            reverse["Transform/ data"][...] = [1, -2, 3]
        self.assertEqual(self.clean()["checks"]["max_absolute_xyz_interface_difference"], [0, 0, 0])

    def test_bad_coordinate_shape_and_nonfinite_coordinates_are_rejected(self):
        with h5py.File(self.source, "r+") as handle:
            handle["/Base/B/GridCoordinates/CoordinateZ/ data"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(CLEAN.TopologyError, "Nonfinite"):
            self.clean()
        fixture_path = self.root / "shape.h5"
        fixture(fixture_path)
        with h5py.File(fixture_path, "r+") as handle:
            del handle["/Base/B/GridCoordinates/CoordinateZ/ data"]
            handle["/Base/B/GridCoordinates/CoordinateZ"].create_dataset(" data", data=np.zeros((2, 2, 3)))
        with self.assertRaisesRegex(CLEAN.TopologyError, "Coordinate shape"):
            self.clean(fixture_path)

    def test_temperature_units_do_not_declare_coordinate_length_units(self):
        with h5py.File(self.source, "r+") as handle:
            temperature = node(handle["/Base"], "Temperature", "DataArray_t", np.array([300.0]))
            units = ["Null", "Null", "Null", "Kelvin", "Null"]
            values = np.frombuffer("".join(value.ljust(32) for value in units).encode("ascii"), dtype="S1").reshape(5, 32)
            node(temperature, "DimensionalUnits", "DimensionalUnits_t", values)
        result = self.clean()
        self.assertEqual(len(result["source_unit_declarations"]), 1)
        self.assertFalse(result["coordinate_units"]["all_coordinate_axes_have_declared_length_units"])
        self.assertEqual(result["coordinate_units"]["declared_length_units"], [])

    def test_undefined_boundary_family_is_rejected(self):
        with h5py.File(self.source, "r+") as handle:
            node(handle["/Base/A/ZoneBC/Wall"], "FamilyName", "FamilyName_t", "missing_family")
        with self.assertRaisesRegex(CLEAN.TopologyError, "Unresolved FamilyName"):
            self.clean()

    def test_check_detects_stale_output_without_rewriting_it(self):
        cases = {"cgns_channel12": "sample.h5"}
        output = self.root / "output"
        argv = ["clean", "--repo-root", str(self.root), "--output", str(output)]
        with patch.object(CLEAN, "CASES", cases), patch("sys.argv", argv):
            CLEAN.main()
        target = output / "cgns_channel12.json"
        target.write_bytes(b"stale\n")
        with patch.object(CLEAN, "CASES", cases), patch("sys.argv", argv + ["--check"]):
            with self.assertRaisesRegex(CLEAN.TopologyError, "missing or stale"):
                CLEAN.main()
        self.assertEqual(target.read_bytes(), b"stale\n")


if __name__ == "__main__":
    unittest.main()
