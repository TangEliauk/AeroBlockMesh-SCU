"""PLOT3D interpretation, subedge topology and source-preservation regressions."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("clean_2d_multiblock", ROOT / "scripts/data/clean_2d_multiblock.py")
cleaner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleaner)


def planar_grid(blocks):
    """Blocks are (nu,nv, mapping(u,v)) with source I varying fastest."""
    tokens = [str(len(blocks))]
    tokens.extend(str(size) for nu, nv, _ in blocks for size in (nu, nv))
    for nu, nv, mapping in blocks:
        points = [mapping(u, v) for v in range(nv) for u in range(nu)]
        tokens.extend(str(p[axis]) for axis in range(2) for p in points)
    return ("\n".join(tokens) + "\n").encode("ascii")


def rectangle(x0, x1, y0, y1, nu=3, nv=3, reverse=False):
    return nu, nv, lambda u, v: (x0+(x1-x0)*u/(nu-1),
                                 y0+(y1-y0)*(1-v/(nv-1) if reverse else v/(nv-1)))


def thin_grid(thin_axis=0, perturb=False, second_offset=None):
    shape = [3, 3, 3]
    shape[thin_axis] = 2
    remaining = [axis for axis in range(3) if axis != thin_axis]
    points = []
    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                indices = [i, j, k]
                plane = indices[thin_axis]
                points.append((indices[remaining[0]] + (0.01 if perturb and plane else 0),
                               (0 if plane == 0 else -1) + (second_offset or 0),
                               indices[remaining[1]]))
    tokens = ["1", *map(str, shape)]
    tokens.extend(str(point[axis]) for axis in range(3) for point in points)
    return "\n".join(tokens).encode("ascii")


class CleanerTests(unittest.TestCase):
    def clean(self, payload, grid_format="plot3d_2d"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.dat"
            path.write_bytes(payload)
            document = cleaner.clean_case(path, case_id="fixture", grid_format=grid_format,
                                          source_relative_path="data/reference/grid.dat")
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(document["source"]["sha256"], hashlib.sha256(payload).hexdigest())
            return document

    def test_full_edge_connection_preserves_source_indices_and_reciprocity(self):
        result = self.clean(planar_grid([rectangle(0, 1, 0, 1), rectangle(1, 2, 0, 1)]))
        self.assertEqual(result["checks"]["unique_interfaces"], 1)
        self.assertEqual(result["checks"]["directed_connections"], 2)
        interface = result["interfaces"][0]
        self.assertEqual(interface["node_count"], 3)
        self.assertEqual(interface["sides"][0]["source_index_range"], [[3, 1], [3, 3]])
        self.assertEqual(interface["sides"][1]["source_index_range"], [[1, 1], [1, 3]])
        self.assertEqual(result["blocks"][0]["neighbors"][0]["donor_block_id"], 2)
        self.assertEqual(result["blocks"][1]["neighbors"][0]["donor_block_id"], 1)
        self.assertTrue(all(b["boundary_type"] == "not_interpreted" for b in result["boundaries"]))

    def test_t_junction_has_two_partial_donor_edges_without_duplicate_cells(self):
        result = self.clean(planar_grid([rectangle(0, 1, 0, .5), rectangle(0, 1, .5, 1),
                                         rectangle(1, 2, 0, 1, nv=5)]))
        self.assertEqual(result["checks"]["unique_interfaces"], 3)
        right = result["blocks"][2]
        patches = sorted(n["edge_point_range"] for n in right["neighbors"] if n["edge"] == "U1")
        self.assertEqual(patches, [[1, 3], [3, 5]])

    def test_reversed_index_orientation_is_recorded(self):
        result = self.clean(planar_grid([rectangle(0, 1, 0, 1), rectangle(1, 2, 0, 1, reverse=True)]))
        interface = result["interfaces"][0]
        self.assertEqual(interface["orientation"], -1)
        self.assertEqual(interface["sides"][1]["edge_point_range"], [3, 1])
        self.assertEqual(interface["sides"][1]["source_index_range"], [[1, 3], [1, 1]])

    def test_swapped_logical_axes_can_match_different_edges(self):
        other = (3, 3, lambda u, v: (1+v/2, u/2))
        result = self.clean(planar_grid([rectangle(0, 1, 0, 1), other]))
        interface = result["interfaces"][0]
        self.assertEqual([side["edge"] for side in interface["sides"]], ["UMAX", "V1"])

    def test_corner_contact_does_not_create_an_interface(self):
        result = self.clean(planar_grid([rectangle(0, 1, 0, 1), rectangle(1, 2, 1, 2)]))
        self.assertEqual(result["interfaces"], [])
        self.assertEqual(len(result["boundaries"]), 8)

    def test_mismatched_edge_nodes_are_not_interpolated(self):
        result = self.clean(planar_grid([rectangle(0, 1, 0, 1), rectangle(1, 2, 0, 1, nv=4)]))
        self.assertEqual(result["interfaces"], [])

    def test_tolerance_matches_points_on_neighboring_spatial_buckets(self):
        result = self.clean(planar_grid([rectangle(0, 1, -1, 0), rectangle(1+5e-11, 2, -1, 0)]))
        self.assertEqual(len(result["interfaces"]), 1)
        self.assertLessEqual(result["checks"]["max_absolute_interface_difference"], cleaner.ATOL)

    def test_nonmanifold_duplicate_donor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-manifold"):
            self.clean(planar_grid([rectangle(0, 1, 0, 1), rectangle(1, 2, 0, 1), rectangle(1, 2, 0, 1)]))

    def test_thin_index_axis_first_middle_or_last_is_detected(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                result = self.clean(thin_grid(axis), "plot3d_thin_3d")
                projection = result["blocks"][0]["projection"]
                self.assertEqual(projection["thin_logical_axis"], "IJK"[axis])
                self.assertEqual(projection["thin_physical_axis"], "Y")
                self.assertEqual(projection["physical_axes"], ["X", "Z"])
                self.assertEqual(projection["thin_coordinate_values"], [0, -1])
                self.assertEqual(result["blocks"][0]["planar_dimensions"], [3, 3])
                self.assertTrue(all(e["source_index_range"][0][axis] == 1 for e in result["blocks"][0]["edges"]))

    def test_unequal_thin_planes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "two-plane extrusion"):
            self.clean(thin_grid(perturb=True), "plot3d_thin_3d")

    def test_coordinate_repeat_notation_comma_and_d_exponent(self):
        payload = b"1\n2 2\n0D0, 1D0, 0D0, 1D0,\n2*0D0, 2*1D0\n"
        result = self.clean(payload)
        self.assertEqual(result["blocks"][0]["corners"][2]["point"], [1, 1])

    def test_invalid_repeat_and_null_values_are_rejected(self):
        base = b"1\n2 2\n0 1 0 1 0 0 1 "
        for ending in (b"0*1", b"2*", b"2**1", b"-1*1", b"999999999*1", b",,1"):
            with self.subTest(ending=ending), self.assertRaises(ValueError):
                self.clean(base+ending)

    def test_truncated_extra_nonfinite_and_dimension_errors(self):
        valid = planar_grid([rectangle(0, 1, 0, 1)])
        for payload in (valid.rsplit(b"\n", 2)[0], valid+b"0", valid.replace(b"0.0", b"nan", 1),
                        valid.replace(b"0.0", b"1e999", 1), b"1\n1 2\n0", b"1\n2.0 2\n0"):
            with self.subTest(payload=payload[:30]), self.assertRaises(ValueError):
                self.clean(payload)

    def test_declared_resource_limits_checked_before_coordinate_allocation(self):
        with self.assertRaisesRegex(ValueError, "resource bound"):
            self.clean(b"1\n999999999 999999999\n")

    def test_gzip_integrity_source_immutability_and_deterministic_json(self):
        payload = gzip.compress(planar_grid([rectangle(0, 1, 0, 1)]), mtime=0)
        first, second = self.clean(payload), self.clean(payload)
        self.assertTrue(first["source"]["gzip_crc_checked"])
        self.assertEqual(cleaner.stable_bytes(first), cleaner.stable_bytes(second))
        self.assertFalse(json.loads(cleaner.stable_bytes(first))["has_original_expert_action_history"])
        broken = bytearray(payload)
        broken[-5] ^= 1
        with self.assertRaises((ValueError, OSError, EOFError)):
            self.clean(bytes(broken))

    def test_check_mode_preserves_existing_file_and_rejects_stale_result(self):
        result = self.clean(planar_grid([rectangle(0, 1, 0, 1)]))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"output.json"
            cleaner.write_or_check(result, output, False)
            before = output.read_bytes()
            cleaner.write_or_check(result, output, True)
            self.assertEqual(output.read_bytes(), before)
            output.write_bytes(b"stale user result")
            with self.assertRaisesRegex(ValueError, "differs"):
                cleaner.write_or_check(result, output, True)
            self.assertEqual(output.read_bytes(), b"stale user result")

    def test_explicit_cli_cannot_overwrite_source_or_resolved_relative_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"grid.dat"
            before = planar_grid([rectangle(0, 1, 0, 1)])
            source.write_bytes(before)
            (root/"nested").mkdir()
            for output in (source, root/"nested"/".."/"grid.dat"):
                with self.subTest(output=output), patch.object(sys, "argv", ["cleaner", "--grid", str(source),
                      "--format", "plot3d_2d", "--case-id", "fixture", "--output", str(output)]):
                    with self.assertRaisesRegex(ValueError, "aliases a source"):
                        cleaner.main()
                self.assertEqual(source.read_bytes(), before)

    def test_hardlinked_source_output_alias_is_rejected_without_changing_either_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"grid.dat", Path(tmp)/"output.json"
            before = planar_grid([rectangle(0, 1, 0, 1)])
            source.write_bytes(before)
            os.link(source, output)
            with self.assertRaisesRegex(ValueError, "aliases a source"):
                cleaner.preflight_cases([(source, "fixture", "plot3d_2d", None, output)])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(output.read_bytes(), before)

    def test_preflight_rejects_cross_case_source_collision_before_any_output_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second, first_output = root/"first.dat", root/"second.dat", root/"first.json"
            before = planar_grid([rectangle(0, 1, 0, 1)])
            first.write_bytes(before)
            second.write_bytes(before)
            cases = [(first, "first", "plot3d_2d", None, first_output),
                     (second, "second", "plot3d_2d", None, first)]
            with self.assertRaisesRegex(ValueError, "aliases a source"):
                cleaner.preflight_cases(cases)
            self.assertFalse(first_output.exists())
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(second.read_bytes(), before)

    def test_duplicate_output_and_hardlinked_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"grid.dat"
            source.write_bytes(planar_grid([rectangle(0, 1, 0, 1)]))
            output, alias = root/"first.json", root/"second.json"
            output.write_bytes(b"existing cleaned result")
            os.link(output, alias)
            for second_output in (output, alias):
                with self.subTest(output=second_output), self.assertRaisesRegex(ValueError, "aliased output"):
                    cleaner.preflight_cases([(source, "first", "plot3d_2d", None, output),
                                             (source, "second", "plot3d_2d", None, second_output)])
            self.assertEqual(output.read_bytes(), b"existing cleaned result")

    def test_collection_rejects_wrong_prefix_escape_or_duplicate_output_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/cleaner.RAW_PREFIX/"first.dat"
            source.parent.mkdir(parents=True)
            before = planar_grid([rectangle(0, 1, 0, 1)])
            source.write_bytes(before)
            manifest_path = source.parent/"manifest.json"
            valid = {"case_id": "first", "source_grid": cleaner.RAW_PREFIX+"first.dat",
                     "grid_format": "plot3d_2d", "source_declared_blocks": 1,
                     "cleaned_path": cleaner.CLEAN_PREFIX+"first.json"}
            bad_cases = [dict(valid, cleaned_path=cleaner.RAW_PREFIX+"first.dat"),
                         dict(valid, cleaned_path=cleaner.CLEAN_PREFIX+"../escape.json"),
                         dict(valid, source_grid="elsewhere/first.dat"),
                         dict(valid, case_id="second")]
            for invalid in bad_cases:
                manifest_path.write_text(json.dumps({"cases": [valid, invalid]}), encoding="utf-8")
                with self.subTest(invalid=invalid), patch.object(sys, "argv", ["cleaner", "--repo-root", str(root)]):
                    with self.assertRaises(ValueError):
                        cleaner.main()
                self.assertFalse((root/valid["cleaned_path"]).exists())
                self.assertEqual(source.read_bytes(), before)

    def test_later_parse_failure_does_not_partially_write_valid_earlier_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_folder = root/cleaner.RAW_PREFIX
            source_folder.mkdir(parents=True)
            first, second = source_folder/"first.dat", source_folder/"second.dat"
            first.write_bytes(planar_grid([rectangle(0, 1, 0, 1)]))
            second.write_bytes(b"1 3 3 0")
            cases = [{"case_id": name, "source_grid": cleaner.RAW_PREFIX+name+".dat",
                      "cleaned_path": cleaner.CLEAN_PREFIX+name+".json", "grid_format": "plot3d_2d",
                      "source_declared_blocks": 1} for name in ("first", "second")]
            (source_folder/"manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
            with patch.object(sys, "argv", ["cleaner", "--repo-root", str(root)]):
                with self.assertRaisesRegex(ValueError, "Truncated"):
                    cleaner.main()
            self.assertFalse((root/cleaner.CLEAN_PREFIX/"first.json").exists())
            self.assertFalse((root/cleaner.CLEAN_PREFIX/"second.json").exists())


if __name__ == "__main__":
    unittest.main()
