"""Regression tests for destructive failure modes and provenance accounting."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from fractions import Fraction

SCRIPT = next(candidate for parent in Path(__file__).resolve().parents
              for candidate in (parent / "scripts" / "data" / "clean_uiuc.py",
                                parent / "scripts" / "clean_uiuc.py") if candidate.is_file())
spec = importlib.util.spec_from_file_location("clean_uiuc", SCRIPT)
clean = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = clean
spec.loader.exec_module(clean)


def toy_points(blunt=False):
    heights = [0, 2, 4, 5, 6, 5, 4, 2, 0]
    top = [(Fraction(i, 8), Fraction(h, 64) + (Fraction(i, 1024) if blunt else 0)) for i, h in enumerate(heights)]
    bottom = [(x, -y) for x, y in top]
    return top[::-1] + bottom[1:]


def encode(points, title="Toy airfoil"):
    return (title + "\n" + "\n".join(f"{float(x):.10f} {float(y):.10f}" for x, y in points) + "\n").encode("ascii")


class GeometryTests(unittest.TestCase):
    def test_comments_dos_eof_and_provenance_without_smoothing(self):
        raw = encode(toy_points()).replace(b"\n", b"\r\n") + b"# comment\r\n\x1a"
        geom, provenance = clean.clean_geometry(raw)
        self.assertEqual(geom["quality"]["vertex_count"], 16)
        self.assertEqual(provenance["quality_flags"]["explicit_closing_vertices_removed"], 1)
        self.assertEqual(provenance["quality_flags"]["raw_normalized_last_to_first_distance"], 0)
        self.assertFalse(provenance["quality_flags"]["smoothing_applied"])
        self.assertGreater(clean.signed_area(geom["polygon"]), 0)

    def test_exact_normalization_duplicates_translation_scale_winding_start(self):
        points = toy_points()[:-1]
        transformed = [(2 * x + 10, 2 * y - 3) for x, y in points]
        transformed = transformed[5:] + transformed[:5]
        transformed = transformed[::-1]
        transformed.append(transformed[0])
        a, _ = clean.clean_geometry(encode(points + [points[0]]))
        b, _ = clean.clean_geometry(encode(transformed))
        self.assertEqual(a["canonical_sha256"], b["canonical_sha256"])
        self.assertEqual(a["polygon"], b["polygon"])

    def test_lednicer_count_header_and_surface_order(self):
        points = toy_points()
        top, bottom = points[:9][::-1], [points[8]] + points[9:]
        raw = encode([(Fraction(9), Fraction(9))] + top + bottom)
        a, provenance = clean.clean_geometry(raw)
        b, _ = clean.clean_geometry(encode(points))
        self.assertEqual(provenance["parse_format"], "lednicer_separate_surfaces")
        self.assertEqual(a["canonical_sha256"], b["canonical_sha256"])

    def test_lednicer_wrong_counts_are_not_silently_repaired(self):
        raw = b"Toy\n9 9\n0 0\n1 0\n"
        with self.assertRaises(clean.Rejected) as context:
            clean.parse_dat(raw)
        self.assertEqual(context.exception.reason, "lednicer_count_mismatch")

    def test_blunt_te_preserved_as_implicit_segment(self):
        geometry, provenance = clean.clean_geometry(encode(toy_points(blunt=True)))
        self.assertEqual(geometry["quality"]["vertex_count"], 17)
        self.assertTrue(provenance["quality_flags"]["open_chain_has_implicit_te_segment"])
        self.assertGreater(provenance["quality_flags"]["raw_normalized_last_to_first_distance"], 0)

    def test_only_exact_consecutive_duplicates_removed(self):
        points = toy_points()
        points.insert(3, points[2])
        _, provenance = clean.clean_geometry(encode(points))
        self.assertEqual(provenance["quality_flags"]["consecutive_exact_duplicates_removed"], 1)

    def test_near_duplicates_are_retained_as_distinct_geometries(self):
        points = toy_points()
        nearby = list(points)
        x, y = nearby[4]
        nearby[4] = (x, y + Fraction(1, 100000000))
        a, _ = clean.clean_geometry(encode(points))
        b, _ = clean.clean_geometry(encode(nearby))
        self.assertNotEqual(a["canonical_sha256"], b["canonical_sha256"])

    def test_nonfinite_and_broken_numeric_rows_rejected(self):
        for raw, reason in [(b"Toy\n0 0\nnan 1\n", "non_finite_coordinate"), (b"Toy\n0 0\nBROKEN\n", "malformed_coordinate_line")]:
            with self.subTest(reason=reason), self.assertRaises(clean.Rejected) as context:
                clean.parse_dat(raw)
            self.assertEqual(context.exception.reason, reason)

    def test_crossings_touches_and_adjacent_exemption(self):
        self.assertIsNotNone(clean.find_intersection([[0, 0], [1, 1], [0, 1], [1, 0]]))
        self.assertIsNone(clean.find_intersection([[0, 0], [1, 0], [1, 1], [0, 1]]))
        self.assertTrue(clean.segments_intersect([0, 0], [1, 0], [.5, 0], [1.5, 0]))

    def test_degenerate_polygon_rejected(self):
        with self.assertRaises(clean.Rejected) as context:
            clean.clean_geometry(encode([(Fraction(i, 16), Fraction(0)) for i in range(17)]))
        self.assertEqual(context.exception.reason, "degenerate_or_tiny_area")

    def test_open_non_te_gap_is_not_filled(self):
        points = toy_points()[:-1]
        points = points[4:] + points[:4]
        with self.assertRaises(clean.Rejected) as context:
            clean.clean_geometry(encode(points))
        self.assertEqual(context.exception.reason, "open_contour_endpoints_away_from_te")

    def test_end_to_end_unique_duplicate_rejected_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, outputs = root / "raw", root / "clean"
            inputs.mkdir()
            first = encode(toy_points())
            (inputs / "a.dat").write_bytes(first)
            (inputs / "copy.dat").write_bytes(first)
            (inputs / "bad.dat").write_bytes(b"Bad\n0 0\n1 NaN\n")
            summary = clean.run(inputs, outputs)
            self.assertEqual((summary["unique_samples"], summary["duplicate_source_files"], summary["rejected_source_files"]), (1, 1, 1))
            sample = json.loads((outputs / "samples.jsonl").read_text())
            self.assertEqual(sample["aliases"], ["a", "copy"])
            self.assertEqual(len(sample["source_files"]), 2)
            self.assertEqual((inputs / "a.dat").read_bytes(), first)
            self.assertNotIn(str(inputs), (outputs / "summary.json").read_text())
            self.assertNotIn(str(inputs), (outputs / "samples.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
