import fs from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [directory, modules, mode] = process.argv.slice(2);
const requireBundle = createRequire(path.resolve(modules, "..", "artifact-entry.cjs"));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(requireBundle.resolve("@oai/artifact-tool")).href);
const input = JSON.parse(await fs.readFile(path.join(directory, "workbook.json"), "utf8"));
const book = Workbook.create();
const font = { name: "Microsoft YaHei", size: 10, color: "#203346" };
const header = { fill: "#173B58", font: { ...font, bold: true, color: "#FFFFFF" }, rowHeight: 26 };

for (const spec of input.sheets) {
  const sheet = book.worksheets.add(spec.name);
  sheet.showGridLines = false;
  const width = spec.headers.length;
  const end = spec.rows.length + 4;
  sheet.getRangeByIndexes(0, 0, 1, width).merge();
  sheet.getCell(0, 0).values = [[spec.title]];
  sheet.getRangeByIndexes(0, 0, 1, width).format = {
    fill: "#0D2940", font: { ...font, size: 15, bold: true, color: "#FFFFFF" }, rowHeight: 33,
  };
  sheet.getRangeByIndexes(1, 0, 1, width).merge();
  sheet.getCell(1, 0).values = [[spec.note]];
  sheet.getRangeByIndexes(1, 0, 1, width).format = {
    font: { ...font, color: "#365A72" }, fill: "#EAF2F6", rowHeight: 27,
  };
  const headers = sheet.getRangeByIndexes(3, 0, 1, width);
  headers.values = [spec.headers];
  headers.format = header;
  for (let c = 0; c < width; c++) {
    sheet.getRangeByIndexes(0, c, end, 1).format.columnWidth = spec.widths[c];
  }
  for (let start = 0; start < spec.rows.length; start += 250) {
    const rows = spec.rows.slice(start, start + 250);
    const range = sheet.getRangeByIndexes(start + 4, 0, rows.length, width);
    range.values = rows;
  }
  if (spec.rows.length > 0) {
    // A native table styles the case rows without materializing a format for every cell.
    if (spec.id !== "cases") {
      const body = sheet.getRangeByIndexes(4, 0, spec.rows.length, width);
      body.format.font = font;
      body.format.rowHeight = 19;
    }
    for (const [col, format] of (spec.formats ?? [])) {
      sheet.getRangeByIndexes(4, col, spec.rows.length, 1).setNumberFormat(format);
    }
    if (spec.ratioColumn !== undefined) {
      sheet.getRangeByIndexes(4, spec.ratioColumn, spec.rows.length, 1).formulas =
        spec.rows.map((_, i) => ["=E" + (i+5) + "/D" + (i+5)]);
    }
    sheet.tables.add(sheet.getRangeByIndexes(3, 0, spec.rows.length + 1, width).address.split("!").at(-1),
      true, "Fusion_" + spec.id).style = "TableStyleMedium2";
  }
  sheet.freezePanes.freezeRows(4);
  spec.rows.length = 0;
}
book.recalculate();
if (mode !== "--fragment") {
  const report = await book.inspect({ kind: "region", sheetId: input.sheets[0].name, range: "A4:J9", maxChars: 4500 });
  const errors = await book.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#N/A",
    options: { useRegex: true, maxResults: 20 }, maxChars: 4000 });
  if (errors.ndjson && /"cell"\s*:/.test(errors.ndjson)) {
    throw new Error(errors.ndjson);
  }
  await fs.writeFile(path.join(directory, "workbook_checks.json"),
    JSON.stringify({ summary: report.ndjson, errorScan: errors.ndjson }, null, 2));
}
const xlsx = await SpreadsheetFile.exportXlsx(book);
await xlsx.save(path.join(directory, "实验结果.xlsx"));
if (mode !== "--fragment") {
  for (const spec of input.sheets) {
    const preview = await book.render({ sheetName: spec.name, range: spec.preview, scale: 1.2, format: "png" });
    await fs.writeFile(path.join(directory, "sheet_" + spec.id + ".png"), new Uint8Array(await preview.arrayBuffer()));
  }
}
