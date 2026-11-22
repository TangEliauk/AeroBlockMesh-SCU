"""Regression checks for source parsers and interface integrity."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/data/clean_engineering_multiblock.py"
SPEC = importlib.util.spec_from_file_location("clean_engineering_multiblock", SCRIPT)
cleaner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleaner)


def icem_fixture():
    nodes = "\n".join(f"{i % 2} {(i // 2) % 2} {i // 4} 3 0 {i} x 0 {{ {i % 2} {(i // 2) % 2} {i // 4} }}"
                      for i in range(8))
    return ("# blocking file written by ANSYS ICEM CFD Mesh Editor 5.0 pre\n"
            "FAMILIES\n{\nFLUID 1\n}\n"
            "PREPOINTS 1 internal_ids\n{\n100 0 0 0 CAD_POINT\n}\n"
            "NODES new_numbering\n{\n" + nodes + "\n}\n"
            "GRID 0 2\n{\n0 2\n1 2\n}\n"
            "GRID 1 2\n{\n0 2\n1 2\n}\n"
            "GRID 2 2\n{\n0 2\n1 2\n}\n"
            "SUPERBLOCKS\n{\n{ 42 2 2 2 -1 0:0:0 1:0:0 2:0:0 { 0 1 2 3 4 5 6 7 } 99 0 0 }\n}\n"
            "UNRECOGNIZED_SETTING 8\nreturn\n")


def gman_fixture():
    lines = ["1", "file run.cgd", "units feet"]
    for zone in (1, 2):
        lines.append(f"ZONE {zone}")
        for face in cleaner.FACES:
            lines.append(f"BOUNDARY {face}")
            if (zone, face) == (1, "IMAX"):
                lines.append("COUPLED TO ZONE 2 BOUNDARY I1")
            elif (zone, face) == (2, "I1"):
                lines.append("COUPLED TO ZONE 1 BOUNDARY IMAX")
            else:
                lines.append("FREESTREAM")
            lines.append("UPDATE")
    return "\n".join(lines)


class IcemParserTests(unittest.TestCase):
    def parse(self, text=None):
        return cleaner.parse_icem_blk((text or icem_fixture()).encode("ascii"), "fixture.blk")

    def test_geometry_namespace_is_separate_and_opaque_fields_survive(self):
        result = self.parse()
        self.assertEqual(result["prepoints"][0]["id"], 100)
        block = result["blocks"][0]
        self.assertEqual(block["node_ids_source_order"], list(range(8)))
        self.assertEqual(block["opaque_trailing_attributes"], [99, 0, 0])
        self.assertEqual(block["bounds"], {"min": [0, 0, 0], "max": [1, 1, 1]})
        self.assertEqual(result["unparsed_sections"][0]["header"], "UNRECOGNIZED_SETTING 8")

    def test_unknown_node_attributes_and_projection_are_not_reinterpreted(self):
        text = icem_fixture().replace("0 0 0 3 0 0 x 0 { 0 0 0 }",
                                      "0 0 0 3n 0 0 x 0 { 0 0 0 4:1 } - 2:0.5")
        node = self.parse(text)["nodes"][0]
        self.assertEqual(node["opaque_source_attributes"], ["3n", "0", "0"])
        self.assertEqual(node["opaque_extra_logical_references_as_stored"], ["4:1"])
        self.assertEqual(node["opaque_projection_as_stored"], "- 2:0.5")

    def test_bad_header_rejected(self):
        with self.assertRaises(ValueError):
            self.parse(icem_fixture().replace("ANSYS ICEM CFD Mesh Editor", "other format"))

    def test_declared_prepoint_count_must_match(self):
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.parse(icem_fixture().replace("PREPOINTS 1", "PREPOINTS 2"))

    def test_nonfinite_prepoint_and_node_rejected(self):
        for text in (icem_fixture().replace("100 0 0 0", "100 nan 0 0"),
                     icem_fixture().replace("0 0 0 3 0 0 x", "inf 0 0 3 0 0 x")):
            with self.subTest(text=text[:40]), self.assertRaisesRegex(ValueError, "Non-finite"):
                self.parse(text)

    def test_missing_corner_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, "corner references unknown"):
            self.parse(icem_fixture().replace("{ 0 1 2 3 4 5 6 7 }", "{ 0 1 2 3 4 5 6 999 }"))

    def test_node_reference_count_must_match_lattice_shape(self):
        with self.assertRaisesRegex(ValueError, "node-reference count"):
            self.parse(icem_fixture().replace("{ 0 1 2 3 4 5 6 7 }", "{ 0 1 2 3 4 5 6 }"))

    def test_unknown_grid_interval_rejected(self):
        with self.assertRaisesRegex(ValueError, "GRID interval"):
            self.parse(icem_fixture().replace("0:0:0 1:0:0", "0:9:0 1:0:0"))

    def test_duplicate_node_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate or negative node ID"):
            self.parse(icem_fixture().replace("3 0 7 x 0", "3 0 6 x 0"))

    def test_missing_proprietary_mid_nodes_are_reported_without_invention(self):
        text = icem_fixture().replace("GRID 1 2\n{\n0 2\n1 2\n}",
                                      "GRID 1 3\n{\n0 2\n0.5 2\n1 2\n}")
        text = text.replace("42 2 2 2 -1 0:0:0 1:0:0 2:0:0 { 0 1 2 3 4 5 6 7 }",
                            "42 2 3 2 -1 0:0:0 1:0:0 1:1:0 2:0:0 { 0 1 800 801 2 3 4 5 802 803 6 7 }")
        result = self.parse(text)
        self.assertEqual(result["blocks"][0]["unresolved_noncorner_node_ids"], [800, 801, 802, 803])
        self.assertFalse(result["checks"]["all_node_references_resolve"])
        self.assertTrue(result["checks"]["all_corner_node_references_resolve"])
        self.assertEqual(len(result["nodes"]), 8)

    def test_unbalanced_section_and_unparsed_block_content_rejected(self):
        for text in (icem_fixture().replace("\n}\nUNRECOGNIZED_SETTING", "\nUNRECOGNIZED_SETTING"),
                     icem_fixture().replace("SUPERBLOCKS\n{", "SUPERBLOCKS\n{\njunk")):
            with self.subTest(text=text[-90:]), self.assertRaises(ValueError):
                self.parse(text)


class Plot3dParserTests(unittest.TestCase):
    def test_repeat_notation_d_exponent_and_i_fastest_order(self):
        # One 2x2x2 block; i coordinates alternate, then four zero/four one j.
        text = "1\n2,2,2\n0 1 0 1 0 1 0 1\n2*0 2*1 2*0 2*1\n4*0.D+0 4*1.D+0"
        block = cleaner.parse_plot3d(text)[0]
        self.assertEqual(block["node_count"], 8)
        self.assertEqual(block["corners"][1]["coordinates"], [1, 0, 0])
        self.assertEqual(block["corners"][-1]["coordinates"], [1, 1, 1])
        self.assertEqual(cleaner.face_points(block, "IMAX")[0], [2, 2])

    def test_bad_header_dimensions_and_coordinate_counts(self):
        for text in ("0", "2\n2 2 2", "1\n2.0 2 2\n24*0", "1\n1 2 2\n12*0",
                     "1\n2 2 2\n23*0", "1\n2 2 2\n25*0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                cleaner.parse_plot3d(text)

    def test_nonfinite_and_invalid_repeats_rejected(self):
        for text in ("1\n2 2 2\n24*nan", "1\n2 2 2\n24*1e999",
                     "1\n2 2 2\n0*0 24*0", "1\n2 2 2\n2**0 22*0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                cleaner.parse_plot3d(text)


class GmanParserTests(unittest.TestCase):
    def test_units_faces_and_reciprocal_coupling(self):
        faces, declarations = cleaner.parse_gman(gman_fixture(), {1, 2})
        self.assertEqual(declarations["units_declaration"], "feet")
        self.assertEqual(len(faces), 12)
        self.assertEqual(sum(face["type"] == "COUPLED" for face in faces), 2)

    def test_nonreciprocal_and_unknown_zone_couplings_rejected(self):
        for text in (gman_fixture().replace("COUPLED TO ZONE 1 BOUNDARY IMAX", "FREESTREAM"),
                     gman_fixture().replace("COUPLED TO ZONE 2", "COUPLED TO ZONE 3")):
            with self.subTest(text=text[-90:]), self.assertRaisesRegex(ValueError, "not reciprocal"):
                cleaner.parse_gman(text, {1, 2})

    def test_duplicate_faces_and_missing_updates_rejected(self):
        for text in (gman_fixture().replace("BOUNDARY KMAX", "BOUNDARY K1", 1),
                     gman_fixture().replace("UPDATE", "", 1)):
            with self.subTest(text=text[:90]), self.assertRaises(ValueError):
                cleaner.parse_gman(text, {1, 2})

    def test_missing_face_and_unknown_commands_rejected(self):
        for text in (gman_fixture().replace("BOUNDARY KMAX\nFREESTREAM\nUPDATE", "", 1),
                     gman_fixture() + "\nGUESS A TRANSFORM"):
            with self.subTest(text=text[-90:]), self.assertRaises(ValueError):
                cleaner.parse_gman(text, {1, 2})

    def test_reversed_surface_mapping_is_found_and_mismatch_is_not_hidden(self):
        shape = [2, 3]
        first = [[float(u), float(v), float(u + v)] for v in range(3) for u in range(2)]
        second = [point for v in range(3) for point in reversed(first[2*v:2*v+2])]
        match = cleaner.match_faces(shape, first, shape, second)
        self.assertTrue(match["unique_mapping_verified"])
        self.assertTrue(match["matching_mappings"][0]["reverse_source_u"])
        second[0] = [99, 99, 99]
        self.assertFalse(cleaner.match_faces(shape, first, shape, second)["coordinates_match"])

    def test_unequal_face_node_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, "different node counts"):
            cleaner.match_faces([2, 2], [[0, 0, 0]]*4, [2, 3], [[0, 0, 0]]*6)


if __name__ == "__main__":
    unittest.main()
