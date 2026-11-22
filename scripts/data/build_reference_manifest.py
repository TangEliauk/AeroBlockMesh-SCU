"""Build a portable reference-only inventory from earlier acquisition records."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
from urllib.parse import quote

TMR_LICENSE = "https://github.com/TMBWG/turbmodels/blob/main/LICENSE"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Root of the original acquired research resources")
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root
    assets = []

    def add(asset_id, entry, purpose, limitations, license_name=None, license_url=None,
            url=None, family=None, retrieval=None):
        name = entry.get("file", entry.get("name", entry.get("path", ""))).replace("\\", "/").split("/")[-1]
        # Old index display names are not necessarily the original filename.
        download_url = url or entry.get("url") or entry.get("source_artifact_url")
        if download_url:
            from urllib.parse import urlsplit, unquote
            name = unquote(urlsplit(download_url).path.split("/")[-1])
        if retrieval and retrieval.get("kind") == "zip_member_http_range":
            name = PurePosixPath(retrieval["archive_member"]).name
        item = dict(asset_id=asset_id, file_name=name, download_url=download_url,
                    source_page=entry.get("source_page", entry.get("page")), bytes=entry["bytes"],
                    sha256=entry["sha256"], release_payload="reference_only_not_bundled",
                    geometry_family=family, purpose=purpose, limitations=limitations,
                    retrieval=retrieval or {"kind": "direct"},
                    license_evidence={"repository_or_source_license": license_name,
                                      "evidence_url": license_url or entry.get("source_page", entry.get("page")),
                                      "note": "Source declaration retained; no new blanket MIT license is applied to third-party data."})
        if download_url is None:
            raise ValueError("Missing download URL: " + asset_id)
        val = entry.get("validation", {})
        permitted = ["format_by_signature", "gzip_crc_checked", "nblocks", "dimensions", "points", "quadrilateral_cells", "all_coordinates_finite", "negative_cell_areas", "zero_cell_areas", "wake_max_abs_coordinate_difference", "zip_crc", "file_count"]
        item["acquisition_checks"] = {key: val[key] for key in permitted if key in val}
        if isinstance(val.get("zone_connectivity_validation"), dict):
            permitted2 = ["zone_count", "structured_zone_count", "directed_one_to_one_connections", "unique_one_to_one_interfaces", "max_absolute_xyz_interface_difference"]
            item["acquisition_checks"].update({key: val["zone_connectivity_validation"][key] for key in permitted2 if key in val["zone_connectivity_validation"]})
        if entry.get("declared_zones") is not None:
            item["source_declared_zones"] = entry["declared_zones"]
        assets.append(item)

    cgns = read_json(root / "公开数据集/02_CGNS多块样例/download_record.json")
    for entry in cgns["downloads"]:
        add(entry["id"], entry, entry["uses"], entry["limitations"], family="crm" if entry["id"].startswith("crm") else ("dlr_f6" if "f6" in entry["id"] else "channel"))

    naca = read_json(root / "公开数据集/03_NACA0012基准/download_record.json")
    tail = (root / "公开数据集/03_NACA0012基准/zip_tail_64k.bin").read_bytes()
    archive_bytes = naca["archive"]["total_bytes"]
    eocd = struct.unpack_from("<4s4H2LH", tail, tail.rfind(b"PK\x05\x06"))
    central_size, central_offset = eocd[5:7]
    start = central_offset - (archive_bytes - len(tail))
    directory = tail[start:start + central_size]
    member_info = {}
    cursor = 0
    while cursor < len(directory):
        h = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
        if h[0] != b"PK\x01\x02":
            raise ValueError("Bad original ZIP central directory")
        name_len, extra_len, comment_len = h[10:13]
        name = directory[cursor + 46:cursor + 46 + name_len].decode("utf-8")
        member_info[name] = dict(zip_local_header_offset=h[16], compressed_bytes=h[8], zip_compression_method=h[4], zip_crc32=f"{h[7]:08x}")
        cursor += 46 + name_len + extra_len + comment_len
    for entry in naca["items"]:
        name = entry["path"].split("/")[-1]
        if "archive_member" in entry:
            level = re.search(r"familyI\.(\d)\.", name).group(1)
            retrieval = dict(kind="zip_member_http_range", archive_member=entry["archive_member"], archive_total_bytes=archive_bytes, **member_info[entry["archive_member"]])
            add("naca0012_familyI_level" + level, entry, ["CFD mesh-convergence validation"], ["Single C-grid with wake self-connection; not independent multiblock-topology training examples.", "All density levels belong to one corrected sharp-trailing-edge geometry and must share an evaluation group."], "CC0-1.0 (TMBWG repository)", TMR_LICENSE, family="naca0012_corrected", retrieval=retrieval)
        elif name.endswith(".nmf"):
            level = re.search(r"familyI\.(\d)\.", name).group(1)
            add("naca0012_neutral_map_level" + level, entry, ["Boundary and wake-connection indexing reference"], ["Official NMF is for 3D two-plane extrusion; project indices explicitly for the 2D PLOT3D grid."], "CC0-1.0 (TMBWG repository)", TMR_LICENSE, family="naca0012_corrected")
        elif name == "n0012points_superbig_clust_fix.dat":
            add("naca0012_corrected_surface", entry, ["Corrected scaled sharp-trailing-edge geometry definition"], ["Keep separate from original finite trailing-edge NACA0012 definitions."], "CC0-1.0 (TMBWG repository)", TMR_LICENSE, family="naca0012_corrected")

    m6 = read_json(root / "公开数据集/04_ONERA_M6基准/download_record.json")
    mapping = {"AileM6_with_sharp_TE.stp": "onera_m6_sharp_cad", "AileM6_with_thick_TE.igs": "onera_m6_thick_cad",
               "profile_ONERA-D.dat": "onera_m6_onera_d_section", "profile_M6_streamwise_alongy=0.dat": "onera_m6_root_section",
               "ONERA_M6_Test_Case_TMR.pdf": "onera_m6_geometry_description", "wing-release-072319.zip": "onera_m6_grid_generator",
               "input.nml_wing_strct_stt": "onera_m6_generator_input1", "input.nml_wing_strct_2_stt": "onera_m6_generator_input2", "input_coarsen.nml_wing_strct_stt": "onera_m6_coarsening_input",
               "WIND_reference_run.dat": "onera_m6_wind_run_configuration", "m6wing.x.fmt": "onera_m6_four_block_grid", "m6-gman.a.txt": "onera_m6_gman_boundaries"}
    for entry in m6["items"]:
        if entry["name"] in mapping:
            nasa_glenn = entry["name"] in ["WIND_reference_run.dat", "m6wing.x.fmt", "m6-gman.a.txt"]
            add(mapping[entry["name"]], entry, [entry["purpose"]], ["Geometry versions, units and near-wall treatment must be matched explicitly.", "No expert state-action history is provided."], None if nasa_glenn else "CC0-1.0 (TMBWG repository)", None if nasa_glenn else TMR_LICENSE, family="onera_m6")

    existing = read_json(root / "公开数据集/05_已有资源索引/download_record.json")
    repos = {entry["repo"]: entry for entry in read_json(root / "参考GitHub项目/下载清单.json")["items"]}
    for index, entry in enumerate(existing["items"]):
        if index == 0:
            add("cgns_multielement_airfoil_4", entry, [entry["purpose"]], entry["limitations"],
                url="https://cgns.org/current/examples.html", family="multielement_airfoil")
        elif index == 1:
            add("dpw3_f6_icem_rev2", entry, [entry["purpose"]], entry["limitations"], family="dlr_f6")
        elif index in (7, 8):
            repo = repos["ArjunNarayanan/par-quad"]
            rel = "data/par_test_set.pkl" if index == 7 else "envs/solved_instances.py"
            add("par_quad_test_set" if index == 7 else "par_quad_synthetic_demo_generator", entry, [entry["purpose"]], entry["limitations"], "MIT (repository)", "https://github.com/ArjunNarayanan/par-quad/blob/" + repo["commit"] + "/LICENSE", url="https://raw.githubusercontent.com/ArjunNarayanan/par-quad/" + repo["commit"] + "/" + rel, family="synthetic_polygons")
        elif index in (9, 10):
            repo = repos["CMU-CBML/SRL-AssistedAFM"]
            rel = "SL dataset.zip" if index == 9 else "SL dataset2.zip"
            add("srl_afm_meshes1" if index == 9 else "srl_afm_meshes2", entry, [entry["purpose"]], entry["limitations"], "BSD-3-Clause (repository)", "https://github.com/CMU-CBML/SRL-AssistedAFM/blob/" + repo["commit"] + "/LICENSE", url="https://raw.githubusercontent.com/CMU-CBML/SRL-AssistedAFM/" + repo["commit"] + "/" + quote(rel), family="unstructured_quad_meshes")
    # Resolve the previously acquired four-block CGNS URL from the official page,
    # rather than assuming a Sphinx hashed download path remains stable.
    for asset in assets:
        if asset["asset_id"] == "cgns_multielement_airfoil_4":
            asset["file_name"] = "multi_hdf5.cgns.gz"
            asset["retrieval"] = {"kind": "official_page_link", "link_file_name": "multi_hdf5.cgns.gz"}
            asset["download_url"] = "https://cgns.org/current/examples.html"
    ids = [item["asset_id"] for item in assets]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate asset identifier")
    result = {"schema_version": "1.0", "release_date": "2026-11-22", "assets": assets,
              "asset_count": len(assets), "bundled_large_grid_payloads": 0,
              "requires_construction": ["Aerodynamic expert blocking trajectories with state, legal-action mask, chosen action and next state", "CFD solver configurations and same-condition convergence runs", "Independent geometry-family train/validation/test groups", "Human preparation, repair and handover time measurements"],
              "boundary": "Reference inventory only; terminal grids, synthetic generators and experimental observations are not relabeled as real expert trajectories."}
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", serialized):
        raise ValueError("Absolute Windows path leaked into portable manifest")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(serialized, encoding="utf-8")
    print("portable reference assets", len(assets))


if __name__ == "__main__":
    main()
