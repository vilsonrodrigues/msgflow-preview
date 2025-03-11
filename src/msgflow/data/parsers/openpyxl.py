from msgflow.data.parsers.base import BaseParser
from msgflow.data.parsers.types import XlsxParser

from openpyxl import load_workbook

def find_tables(sheet):
    tables = []
    in_table = False
    start_row = 0
 
    for row in range(1, sheet.max_row + 1):
        # Verifica se a linha está vazia
        row_is_empty = all(cell.value is None for cell in sheet[row])
        if not row_is_empty and not in_table:
            # Início de uma nova tabela
            in_table = True
            start_row = row
        elif row_is_empty and in_table:
            # Fim da tabela atual
            in_table = False
            tables.append((start_row, row - 1))
    
    # Adiciona a última tabela, se estiver incompleta
    if in_table:
        tables.append((start_row, sheet.max_row))
    
    return tables

def extract_table_data(sheet, start_row, end_row):
    table_data = []
    # Determina o número máximo de colunas na tabela
    max_col = max((cell.column for row in range(start_row, end_row + 1) 
                   for cell in sheet[row] if cell.value is not None), default=1)
    
    for row in range(start_row, end_row + 1):
        row_data = [sheet.cell(row, col).value for col in range(1, max_col + 1)]
        table_data.append(row_data)
    return table_data
 

def get_cell_value(sheet, row, col):
    cell = sheet.cell(row, col)
    for merged_range in sheet.merged_cells.ranges:
        if cell.coordinate in merged_range:
            return sheet.cell(merged_range.min_row, merged_range.min_col).value
    return cell.value
 
def extract_table_data(sheet, start_row, end_row):
    table_data = []
    max_col = max((cell.column for row in range(start_row, end_row + 1) 
                   for cell in sheet[row] if cell.value is not None), default=1)
    
    for row in range(start_row, end_row + 1):
        row_data = [get_cell_value(sheet, row, col) for col in range(1, max_col + 1)]
        table_data.append(row_data)
    return table_data
 
 
def table_to_markdown(table_data):
    if not table_data:
        return ""
    
    # Primeira linha é o cabeçalho
    header = table_data[0]
    rows = table_data[1:]
    
    # Linha do cabeçalho
    header_line = "| " + " | ".join(str(cell) if cell is not None else "" for cell in header) + " |"
    # Linha de separação
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    # Linhas de dados
    data_lines = ["| " + " | ".join(str(cell) if cell is not None else "" for cell in row) + " |" 
                  for row in rows]
    
    return "\n".join([header_line, separator_line] + data_lines)
 
def table_to_html(table_data):
    if not table_data:
        return ""
    
    # Primeira linha é o cabeçalho
    header = table_data[0]
    rows = table_data[1:]
    
    # Iniciar a tabela HTML
    html = '<table>\n'
    
    # Adicionar a linha do cabeçalho
    html += '  <tr>\n'
    for cell in header:
        html += f'    <th>{str(cell) if cell is not None else ""}</th>\n'
    html += '  </tr>\n'
    
    # Adicionar as linhas de dados
    for row in rows:
        html += '  <tr>\n'
        for cell in row:
            html += f'    <td>{str(cell) if cell is not None else ""}</td>\n'
        html += '  </tr>\n'
    
    # Fechar a tabela
    html += '</table>'
    
    return html

def xlsx_parser(path): 

    workbook = load_workbook(path, data_only=True)
    
    md_content = ""

    images_dict = {}
    
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        md_content += f"<!-- Sheet: {sheet_name} -->\n"

        for idx, image in enumerate(sheet._images):
            filename = f"{sheet_name}_image_{idx}.png"
            md_content += f"\n![Image on Sheet {sheet_name}: {idx}]({filename})"
            images_dict[filename] = image._data()
        
        tables = find_tables(sheet)
        for i, (start_row, end_row) in enumerate(tables):
            table_data = extract_table_data(sheet, start_row, end_row)
            markdown_table = table_to_markdown(table_data)
            md_content += f"\n\n<!-- Table number: {i + 1}. Position: ({start_row}, {end_row}) -->" 
            md_content += f"\n{markdown_table}\n"

    return {
        "text": md_content.strip(),
        "images": images_dict
    }
