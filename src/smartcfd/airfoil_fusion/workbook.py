"""Portable Excel export for the audited fusion report."""


def write_workbook(path, tables):
    import xlsxwriter

    options = {"strings_to_formulas": False, "strings_to_urls": False}
    with xlsxwriter.Workbook(path, options) as workbook:
        title = workbook.add_format({
            "bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#0D273E"
        })
        note = workbook.add_format({"font_color": "#36546A", "bg_color": "#E7EFF4"})
        header = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#15364D"})
        for table in tables:
            sheet = workbook.add_worksheet(table["name"])
            headers, rows = table["headers"], table["rows"]
            last_column = len(headers) - 1
            sheet.merge_range(0, 0, 0, last_column, table["title"], title)
            sheet.set_row(0, 25)
            sheet.merge_range(1, 0, 1, last_column, table["note"], note)
            sheet.freeze_panes(4, 0)
            formats = {index: workbook.add_format({"num_format": pattern}) for index, pattern in table["formats"]}
            for index, width in enumerate(table["widths"]):
                sheet.set_column(index, index, width, formats.get(index))
            if rows:
                sheet.add_table(3, 0, 3 + len(rows), last_column, {
                    "name": f"Fusion_{table['id']}",
                    "style": "Table Style Light 9",
                    "columns": [{"header": name, "header_format": header} for name in headers],
                })
            else:
                sheet.write_row(3, 0, headers, header)
            ratio_column = table.get("ratioColumn")
            for row_number, row in enumerate(rows, 4):
                sheet.write_row(row_number, 0, row)
                if ratio_column is not None:
                    excel_row = row_number + 1
                    sheet.write_formula(
                        row_number, ratio_column, f"=E{excel_row}/D{excel_row}",
                        formats[ratio_column], row[4] / row[3],
                    )
