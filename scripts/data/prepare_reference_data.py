"""Normalize four TMR experimental tables, preserving conditions and source rows.

Only Python's standard library is required. Input may be a flat raw-data directory
or the original research workspace; output paths never contain machine paths.
"""
import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import urllib.request

TMR_LICENSE = "https://github.com/TMBWG/turbmodels/blob/main/LICENSE"
ETA = {1: "0.2", 2: "0.44", 3: "0.65", 4: "0.8", 5: "0.9", 6: "0.96", 7: "0.99"}
SOURCES = {
    "naca0012_ladson_force": {
        "raw_name": "CLCD_Ladson_expdata.dat",
        "aliases": ["NACA0012-CLCD_Ladson_expdata.dat"],
        "url": "https://tmbwg.github.io/turbmodels/NACA0012_validation/CLCD_Ladson_expdata.dat",
        "page": "https://tmbwg.github.io/turbmodels/naca0012_val.html",
        "expected_sha256": "78cd2f6aa4968e80f44cbf6c96f699bd9c6e45681d958ec528a10cf72ed23357",
        "columns": ["dataset_id", "grit_number", "alpha_deg", "cl", "cd", "source_line"],
        "primary_key": ["grit_number", "alpha_deg"],
        "units": {"grit_number": "grit designation; not a physical length", "alpha_deg": "degree", "cl": "1", "cd": "1", "source_line": "1-based line index"},
        "expected_group_counts": {"80": 17, "120": 18, "180": 18},
        "conditions": {"mach": 0.15, "reynolds": 6000000, "reynolds_reference": "airfoil chord", "transition": "tripped", "roughness_grit_groups": [80, 120, 180]},
        "attribution": "C. L. Ladson, NASA TM 4074, 1988; force-data transcription distributed by TMR.",
        "caveats": ["Do not pool grit groups or identify them as independent geometries.", "Post-stall observations and high drag values are retained, not deleted as outliers."],
    },
    "naca0012_gregory_cp": {
        "raw_name": "CP_Gregory_expdata.dat",
        "aliases": ["NACA0012-CP_Gregory_expdata.dat"],
        "url": "https://tmbwg.github.io/turbmodels/NACA0012_validation/CP_Gregory_expdata.dat",
        "page": "https://tmbwg.github.io/turbmodels/naca0012_val.html",
        "expected_sha256": "38d8474bc67661c3cef5e4cd9eeddb95d4a436d06abc1f58a8518066f510647c",
        "columns": ["dataset_id", "alpha_deg", "x_over_c", "cp", "source_line"],
        "primary_key": ["alpha_deg", "x_over_c"],
        "units": {"alpha_deg": "degree", "x_over_c": "1", "cp": "1", "source_line": "1-based line index"},
        "expected_group_counts": {"0": 25, "10": 24, "15": 24},
        "conditions": {"mach": None, "mach_note": "not specified in this digitized source file; not filled from the Ladson experiment", "reynolds": 2880000, "reynolds_reference": "airfoil chord", "alpha_deg_groups": [0, 10, 15], "surface": "upper only", "measurement_form": "digitized from photocopy; approximate"},
        "attribution": "N. Gregory and C. L. O'Reilly, R&M 3726, 1970; digitized upper-surface pressures distributed by TMR.",
        "caveats": ["This is not the Re=6 million Ladson force experiment.", "Small x/c excursions outside [0,1] caused by digitization are retained and flagged, never clipped.", "The raw header calls the report 'NASA R&M 3726'; the report identifier and authors are retained without adopting that organizational label."],
    },
    "onera_m6_case2308_cp": {
        "raw_name": "case_2308.dat", "aliases": ["ONERA-M6-case_2308.dat"],
        "url": "https://tmbwg.github.io/turbmodels/Onerawingnumerics_val/case_2308.dat",
        "page": "https://tmbwg.github.io/turbmodels/onerawingnumerics_val.html",
        "expected_sha256": "020c5fcc58060737024eb87d9404f56bc563f3f6f15e337675c47477fa91f0d0",
        "expected_run": 308, "case_id": 2308, "mach": "0.8395", "alpha_deg": "3.06", "reynolds_millions": "11.72",
    },
    "onera_m6_case2565_cp": {
        "raw_name": "case_2565.dat", "aliases": [],
        "url": "https://tmbwg.github.io/turbmodels/Onerawingnumerics_val/case_2565.dat",
        "page": "https://tmbwg.github.io/turbmodels/onerawingnumerics_val.html",
        "expected_sha256": "6345287f200034492fe258bfa3748920e6b76661727b2eb22f74f1458747f95e",
        "expected_run": 565, "case_id": 2565, "mach": "0.8372", "alpha_deg": "6.06", "reynolds_millions": "11.71",
    },
}
for source in SOURCES.values():
    if "case_id" not in source:
        continue
    source.update(
        columns=["dataset_id", "case_id", "run_id", "section_id", "eta_y_over_semispan", "tap_id", "x_over_l", "z_over_l", "cp", "source_line"],
        primary_key=["section_id", "tap_id"],
        units={"eta_y_over_semispan": "1", "x_over_l": "1", "z_over_l": "1", "cp": "1", "source_line": "1-based line index"},
        expected_group_counts={"1": 34, "2": 34, "3": 34, "4": 34, "5": 45, "6": 45, "7": 45},
        conditions={"mach": float(source["mach"]), "alpha_deg": float(source["alpha_deg"]),
                    "reynolds": int(Decimal(source["reynolds_millions"]) * 1000000),
                    "reynolds_reference": "original experimental blunt-wing mean aerodynamic chord", "mean_aerodynamic_chord_mm": 646.07,
                    "geometry": "original wind-tunnel model with moderately thick trailing edge", "case_id": source["case_id"], "run_id_in_raw_file": source["expected_run"],
                    "section_eta_y_over_semispan": ETA,
                    "separate_cfd_exercise": {"mach": 0.84, "reynolds_root_chord": 14600000, "temperature_rankine": 540, "geometry": "sharp trailing edge", "note": "Separate numerical exercise specification; does not overwrite experimental conditions."}},
        attribution="V. Schmitt and F. Charpin, ONERA-M6 experiment, AGARD AR-138, 1979; numeric pressure tables distributed by TMR.",
        caveats=["Section 6 uses eta=0.96 per TMR correction; some original references list 0.95.", "X/L and Z/L variable names from the raw table are preserved as x_over_l and z_over_l; L is not silently redefined as the root chord.", "Case identifier and raw Run identifier differ; both are retained.", "No upper/lower surface labels are inferred from the sign of Z/L."],
    )


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def number(token):
    value = Decimal(token.replace("D", "E").replace("d", "e"))
    if not value.is_finite():
        raise ValueError("Nonfinite source value")
    return format(value, "f")


def find_input(root, source):
    candidates = []
    for name in [source["raw_name"]] + source["aliases"]:
        candidates.extend([root / name, root / "experiments" / name, root / "参考文献" / "专题补充_2026-11-18" / name,
                           root / "公开数据集" / "04_ONERA_M6基准" / name])
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError("Missing source file: " + source["raw_name"])
    # Multiple known aliases are allowed only when their complete bytes agree.
    if len({sha256(path.read_bytes()) for path in found}) != 1:
        raise ValueError("Conflicting aliases for " + source["raw_name"])
    return found[0]


def parse(dataset_id, source, raw):
    rows, group_counts, declared_counts = [], collections.Counter(), {}
    group, m6_header = None, None
    for line_index, raw_line in enumerate(raw.decode("ascii").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.upper().startswith(("TITLE", "VARIABLES")):
            continue
        if line.upper().startswith("ZONE"):
            if "case_id" in source:
                pattern = r"Run=\s*(\d+),\s*Mach=\s*([.\d]+),\s*Alpha=\s*([.\d]+),\s*Re=\s*([.\d]+)x10\*\*6,\s*Section\s+(\d+)\".*I=\s*(\d+)"
                match = re.search(pattern, line)
                if not match:
                    raise ValueError(f"Unrecognized M6 zone header on line {line_index}")
                run, mach, alpha, reynolds, section, size = match.groups()
                if (int(run) != source["expected_run"] or Decimal(mach) != Decimal(source["mach"])
                    or Decimal(alpha) != Decimal(source["alpha_deg"]) or Decimal(reynolds) != Decimal(source["reynolds_millions"])):
                    raise ValueError("M6 source conditions differ from expected case")
                group = str(int(section))
                declared_counts[group] = int(size)
                m6_header = dict(run_id=int(run), section_id=int(section))
            else:
                title = re.search(r't\s*=\s*"([^"]+)"', line, re.I).group(1)
                group = title.split()[0] if "grit_number" in source["columns"] else title.split("=")[1]
            continue
        tokens = line.split()
        if group is None:
            raise ValueError("Numeric row appeared before a zone header")
        values = [number(token) for token in tokens]
        row = dict(dataset_id=dataset_id, source_line=line_index)
        if "case_id" in source:
            if len(values) != 5:
                raise ValueError("M6 numeric row must contain five values")
            section, tap = int(Decimal(values[0])), int(Decimal(values[1]))
            if Decimal(values[0]) != section or Decimal(values[1]) != tap or section != m6_header["section_id"]:
                raise ValueError("Invalid section/tap identifiers")
            row.update(case_id=source["case_id"], run_id=m6_header["run_id"], section_id=section,
                       eta_y_over_semispan=ETA[section], tap_id=tap,
                       x_over_l=values[2], z_over_l=values[3], cp=values[4])
        elif "grit_number" in source["columns"]:
            if len(values) != 3:
                raise ValueError("Force row must contain three values")
            row.update(grit_number=int(group), alpha_deg=values[0], cl=values[1], cd=values[2])
        else:
            if len(values) != 2:
                raise ValueError("Pressure row must contain two values")
            row.update(alpha_deg=number(group), x_over_c=values[0], cp=values[1])
        rows.append(row)
        group_counts[group] += 1
    if dict(group_counts) != source["expected_group_counts"]:
        raise ValueError(f"Group counts differ from the source contract: {dict(group_counts)}")
    if declared_counts and declared_counts != dict(group_counts):
        raise ValueError("M6 zone I declarations disagree with actual numeric rows")
    keys = [tuple(str(row[column]) for column in source["primary_key"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate observation primary key")
    if not all(set(row) == set(source["columns"]) for row in rows):
        raise ValueError("Missing or unexpected output columns")
    if "grit_number" in source["columns"] and not all(Decimal(row["cd"]) > 0 for row in rows):
        raise ValueError("Nonpositive experimental drag coefficient")
    outside_x = [row["source_line"] for row in rows if "x_over_c" in row and not 0 <= Decimal(row["x_over_c"]) <= 1]
    return rows, dict(group_counts), declared_counts, outside_x


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Directory holding the four raw .dat files; original workspace layout is also supported")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination for experiments/, metadata/ and cleaning_report.json")
    parser.add_argument("--fetch", action="store_true", help="Download raw tables from their official URLs into input-root before normalization")
    parser.add_argument("--verify-upstream", action="store_true", help="Compare complete official response hashes with local raw inputs; no response payload is saved")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        args.input_root.mkdir(parents=True, exist_ok=True)
        for source in SOURCES.values():
            target = args.input_root / source["raw_name"]
            with urllib.request.urlopen(source["url"], timeout=60) as response:
                payload = response.read()
            if sha256(payload) != source["expected_sha256"]:
                raise ValueError("Upstream source changed; review before replacing the release baseline")
            if target.is_file() and target.read_bytes() != payload:
                raise FileExistsError("Will not overwrite conflicting raw input")
            target.write_bytes(payload)
    report = {"schema_version": "1.0", "role": "experimental CFD validation references; not RL training trajectories", "datasets": [], "total_rows": 0,
              "transforms": ["parse Tecplot zone structure", "canonical decimal formatting without rounding", "retain source row ordering and 1-based line provenance", "keep experiments and conditions separate"],
              "dropped_numeric_rows": 0, "imputed_values": 0, "clipped_coordinates": 0, "generated_trajectories": 0}
    for dataset_id, source in SOURCES.items():
        path = find_input(args.input_root, source)
        raw = path.read_bytes()
        actual_sha256 = sha256(raw)
        if actual_sha256 != source["expected_sha256"]:
            raise ValueError("Raw checksum changed: " + source["raw_name"])
        upstream = None
        if args.verify_upstream:
            with urllib.request.urlopen(source["url"], timeout=60) as response:
                upstream = {"url": response.url, "sha256": sha256(response.read())}
            upstream["matches_local_raw"] = upstream["sha256"] == actual_sha256
            if not upstream["matches_local_raw"]:
                raise ValueError("Current official data differs from the local baseline")
        rows, group_counts, declared_counts, outside_x = parse(dataset_id, source, raw)
        csv_path = args.output_root / "experiments" / (dataset_id + ".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=source["columns"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        # Reload output to ensure every observation survived serialization exactly.
        with csv_path.open(encoding="utf-8", newline="") as stream:
            loaded = list(csv.DictReader(stream))
        if loaded != [{key: str(row[key]) for key in source["columns"]} for row in rows]:
            raise ValueError("CSV roundtrip changed the parsed observations")
        metadata = {"schema_version": "1.0", "dataset_id": dataset_id,
                    "csv": "experiments/" + csv_path.name, "row_count": len(rows),
                    "grain": "one measured/digitized observation at one experiment condition and zone",
                    "columns": source["columns"], "units": source["units"], "primary_key": source["primary_key"],
                    "conditions": source["conditions"], "group_counts": group_counts,
                    "source": {"file_name": source["raw_name"], "url": source["url"], "page": source["page"],
                               "bytes": len(raw), "sha256": actual_sha256, "attribution": source["attribution"]},
                    "license_evidence": {"upstream_repository_license": "CC0-1.0", "url": TMR_LICENSE,
                                         "scope_note": "TMBWG repository declaration; original experiment attribution retained. Not relabeled MIT."},
                    "caveats": source["caveats"], "upstream_hash_verification": upstream}
        write_json(args.output_root / "metadata" / (dataset_id + ".json"), metadata)
        entry = {"dataset_id": dataset_id, "rows": len(rows), "source_sha256": actual_sha256,
                 "csv_sha256": sha256(csv_path.read_bytes()), "group_counts": group_counts,
                 "zone_declared_counts": declared_counts, "primary_key_unique": True, "finite_numeric_values": True,
                 "csv_roundtrip_exact": True, "all_raw_numeric_rows_retained": True,
                 "x_over_c_outside_unit_interval_retained": len(outside_x), "out_of_range_source_lines": outside_x,
                 "upstream_hash_matches_local": upstream["matches_local_raw"] if upstream else None}
        report["datasets"].append(entry)
        report["total_rows"] += len(rows)
        print(dataset_id, len(rows), group_counts, flush=True)
    if report["total_rows"] != 668:
        raise ValueError("Unexpected total observation count")
    write_json(args.output_root / "cleaning_report.json", report)


if __name__ == "__main__":
    main()
