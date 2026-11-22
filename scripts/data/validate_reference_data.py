"""Validate published experimental tables without original-machine files."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.data_root / "cleaning_report.json").read_text(encoding="utf-8"))
    rows_by_id = {}
    for check in report["datasets"]:
        dataset_id = check["dataset_id"]
        meta = json.loads((args.data_root / "metadata" / (dataset_id + ".json")).read_text(encoding="utf-8"))
        payload = (args.data_root / meta["csv"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == check["csv_sha256"]
        with (args.data_root / meta["csv"]).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == meta["columns"]
            rows = list(reader)
        assert len(rows) == meta["row_count"] == check["rows"]
        assert len({tuple(row[key] for key in meta["primary_key"]) for row in rows}) == len(rows)
        assert all(Decimal(value).is_finite() for row in rows for key, value in row.items() if key != "dataset_id")
        assert all(row["dataset_id"] == dataset_id for row in rows)
        rows_by_id[dataset_id] = rows
    force = rows_by_id["naca0012_ladson_force"]
    assert Decimal(force[0]["cl"]) == Decimal("-0.4417")
    assert Decimal(force[-1]["cd"]) == Decimal("0.43446")  # retain the post-stall observation
    pressure = rows_by_id["naca0012_gregory_cp"]
    assert sum(not 0 <= Decimal(row["x_over_c"]) <= 1 for row in pressure) == 3
    assert min(Decimal(row["cp"]) for row in pressure) == Decimal("-10.1789")
    for dataset_id, expected_mach, expected_re in [("onera_m6_case2308_cp", 0.8395, 11720000), ("onera_m6_case2565_cp", 0.8372, 11710000)]:
        meta = json.loads((args.data_root / "metadata" / (dataset_id + ".json")).read_text(encoding="utf-8"))
        assert meta["conditions"]["mach"] == expected_mach
        assert meta["conditions"]["reynolds"] == expected_re
        assert all(Decimal(row["eta_y_over_semispan"]) == Decimal("0.96") for row in rows_by_id[dataset_id] if row["section_id"] == "6")
        assert len(rows_by_id[dataset_id]) == 271
    assert sum(map(len, rows_by_id.values())) == report["total_rows"] == 668
    for path in args.data_root.rglob("*.json"):
        assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", path.read_text(encoding="utf-8")), path.name
    print("PASS: four datasets, 668 observations, separate conditions, preserved edge observations, hashes, keys, units and portable metadata")


if __name__ == "__main__":
    main()
