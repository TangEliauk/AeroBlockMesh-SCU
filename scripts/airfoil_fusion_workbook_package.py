"""Package Artifact Tool worksheet fragments without changing their cell values."""

from copy import deepcopy
import json
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile


ROWS_PER_FRAGMENT = 2500
CASE_SHEET = "xl/worksheets/sheet3.xml"
CASE_RELATIONSHIPS = "xl/worksheets/_rels/sheet3.xml.rels"
NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = "{" + NAMESPACE + "}"
XLSX = "实验结果.xlsx"


def _case_table(base):
    relationships = ElementTree.fromstring(base.read(CASE_RELATIONSHIPS))
    target, = [link.attrib["Target"] for link in relationships if link.attrib["Type"].endswith("/table")]
    member = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl/worksheets", target))
    table = ElementTree.fromstring(base.read(member))
    if table.attrib["name"] != "Fusion_cases":
        raise ValueError("The third worksheet does not use the expected case table")
    return member, table


def _append_case_rows(zipped, fragment, styles, offset):
    with ZipFile(fragment) as sheet:
        if sheet.read("xl/styles.xml") != styles:
            raise ValueError("Workbook fragment styles differ")
        document = ElementTree.fromstring(sheet.read(CASE_SHEET))
        rows = document.find(NS+"sheetData")
        strings = None
        count = 0
        for row in rows.findall(NS+"row"):
            old_index = int(row.attrib["r"])
            if old_index < 5:
                continue
            count += 1
            if old_index != count+4:
                raise ValueError("Workbook fragment has missing or disordered case rows")
            new_index = str(old_index+offset)
            row.set("r", new_index)
            for cell in row.findall(NS+"c"):
                if cell.find(NS+"f") is not None:
                    raise ValueError("A case-table fragment contains a formula")
                address = cell.attrib["r"]
                letters = address.rstrip("0123456789")
                if address != letters+str(old_index):
                    raise ValueError("Workbook fragment cell address differs from its row")
                cell.set("r", letters+new_index)
                if cell.get("t") == "s":
                    if strings is None:
                        strings = ElementTree.fromstring(sheet.read("xl/sharedStrings.xml")).findall(NS+"si")
                    value = cell.find(NS+"v")
                    text = strings[int(value.text)]
                    cell.remove(value)
                    cell.set("t", "inlineStr")
                    inline = ElementTree.SubElement(cell, NS+"is")
                    inline.extend(deepcopy(list(text)))
            zipped.write(ElementTree.tostring(row, encoding="utf-8"))
    return count


def _package(parts, destination, total_rows, first_rows):
    ElementTree.register_namespace("x", NAMESPACE)
    with ZipFile(parts[0]) as base, ZipFile(destination, "w", compression=ZIP_DEFLATED) as output:
        table_member, table = _case_table(base)
        previous_ref = table.attrib["ref"]
        match = re.fullmatch(r"(A4:[A-Z]+)([0-9]+)", previous_ref)
        if match is None or int(match[2]) != first_rows+4:
            raise ValueError("The base case table does not match the first fragment")
        new_ref = match[1]+str(total_rows+4)
        table.set("ref", new_ref)
        for view in table.findall(NS+"autoFilter"):
            if view.get("ref") == previous_ref:
                view.set("ref", new_ref)
        base_sheet = base.read(CASE_SHEET)
        close = re.search(rb"</(?:[A-Za-z0-9_]+:)?sheetData>", base_sheet)
        if close is None:
            raise ValueError("Missing base case worksheet data")
        prefix = re.sub(rb'(<(?:[A-Za-z0-9_]+:)?dimension\b[^>]*\bref="[A-Z]+[0-9]+:[A-Z]+)[0-9]+(")',
                        lambda match: match[1]+str(total_rows+4).encode("ascii")+match[2], base_sheet[:close.start()])
        for member in base.infolist():
            if member.filename == CASE_SHEET:
                with output.open(member.filename, "w") as zipped:
                    zipped.write(prefix)
                    count = first_rows
                    styles = base.read("xl/styles.xml")
                    for fragment in parts[1:]:
                        count += _append_case_rows(zipped, fragment, styles, count)
                    if count != total_rows:
                        raise ValueError("The packaged sheet lost case rows")
                    zipped.write(base_sheet[close.start():])
            elif member.filename == table_member:
                output.writestr(member.filename, ElementTree.tostring(table, encoding="utf-8", xml_declaration=True))
            else:
                with base.open(member) as old, output.open(member.filename, "w") as new:
                    shutil.copyfileobj(old, new)


def export(directory, node, modules):
    directory = directory.resolve()
    input_path = directory/"workbook.json"
    spec = json.loads(input_path.read_text(encoding="utf-8"))
    case_spec = spec["sheets"][2]
    if case_spec["id"] != "cases":
        raise ValueError("Unexpected workbook sheet order")
    rows = case_spec["rows"]
    builder = Path(__file__).with_name("airfoil_fusion_workbook.mjs")
    if len(rows) <= ROWS_PER_FRAGMENT:
        subprocess.run([node, str(builder), str(directory), modules], check=True, timeout=600)
        return
    with TemporaryDirectory(dir=directory, prefix="xlsx_parts_") as temporary:
        work = Path(temporary).resolve()
        assert work.parent == directory and work.name.startswith("xlsx_parts_")
        parts = []
        for start in range(0, len(rows), ROWS_PER_FRAGMENT):
            folder = work/f"part_{len(parts):03d}"
            folder.mkdir()
            case_spec["rows"] = rows[start:start+ROWS_PER_FRAGMENT]
            (folder/"workbook.json").write_text(json.dumps(spec, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            command = [node, str(builder), str(folder), modules]
            if parts:
                command.append("--fragment")
            subprocess.run(command, check=True, timeout=120)
            parts.append(folder/XLSX)
        result = work/XLSX
        _package(parts, result, len(rows), ROWS_PER_FRAGMENT)
        for path in parts[0].parent.glob("sheet_*.png"):
            shutil.copyfile(path, directory/path.name)
        checks = json.loads((parts[0].parent/"workbook_checks.json").read_text(encoding="utf-8"))
        checks.update(case_rows=len(rows), case_fragments=len(parts))
        (directory/"workbook_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
        result.replace(directory/XLSX)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export fusion workbook fragments using the optional Node backend")
    parser.add_argument("folder", type=Path)
    parser.add_argument("node")
    parser.add_argument("modules")
    options = parser.parse_args()
    export(options.folder, options.node, options.modules)
