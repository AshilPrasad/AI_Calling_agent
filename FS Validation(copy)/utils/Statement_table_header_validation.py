import camelot
import re
import pandas as pd
from utils.pdf_reading import pdf_open




"""Statement Header tables only extracted"""

def statement_table_extraction(pdf_path, df):
    tables_dict = {}

    for _, row in df.iterrows():
        if row['Validation'] != 'Valid':
            continue

        page_number = row['Page_NO']
        header_found = row['Main_Header']

        # Read PDF page based on header
        if header_found == 'Notes to the Financial Statements':
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_number),
                flavor='stream',
                strip_text='\n',
                row_tol=10,
                column_tol=7,
                table_areas=['0,800,500,450']
            )
        else:
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_number),
                flavor='stream',
                strip_text='\n',
                table_areas=['0,800,800,150'],
                row_tol=7,
                edge_tol=55
            )

        if tables.n == 0:
            continue

        selected_table_df = tables[-1].df

        if header_found == 'Notes to the Financial Statements':
            header_row_idx = None
            for idx, row_vals in selected_table_df.iterrows():
                if any('Sl. No.' in str(cell) for cell in row_vals):
                    header_row_idx = idx
                    break

            if header_row_idx is not None:
                selected_table_df.columns = selected_table_df.iloc[header_row_idx]
                selected_table_df = selected_table_df[header_row_idx + 1:].reset_index(drop=True)

                if 'Shareholders' in selected_table_df.columns:
                    total_mask = selected_table_df['Shareholders'].astype(str).str.strip() == "Total"
                    if total_mask.any():
                        first_total_idx = total_mask.idxmax()
                        selected_table_df = selected_table_df.loc[:first_total_idx].reset_index(drop=True)

                last_two_cols = selected_table_df.columns[-2:]
                for col in last_two_cols:
                    selected_table_df[col] = (
                        selected_table_df[col]
                        .astype(str)
                        .str.replace(',', '', regex=False)
                        .str.replace(r'\((.*?)\)', r'-\1', regex=True))
                    
                    selected_table_df[col] = pd.to_numeric(selected_table_df[col], errors='coerce')

            else:
                selected_table_df.columns = selected_table_df.iloc[0]
                selected_table_df = selected_table_df[1:].reset_index(drop=True)

        else:
            # Process non-"Notes" tables
            last_col = selected_table_df.columns[-1]
            total_mask = selected_table_df[last_col].astype(str).str.strip() == "Total"
            if total_mask.any():
                total_row_idx = total_mask.idxmax()
                selected_table_df = selected_table_df.loc[total_row_idx:].reset_index(drop=True)

            selected_table_df.columns = selected_table_df.iloc[0]
            selected_table_df = selected_table_df[1:].reset_index(drop=True)

            for col in selected_table_df.columns[1:]:
                selected_table_df[col] = selected_table_df[col].astype(str).replace(r'\((.*?)\)', r'-\1', regex=True)

            # Handle 4-column tables with specific formatting
            if selected_table_df.shape[1] == 4:
                selected_table_df.columns = ["Description", "Note", "Year", "Previous year"]
                selected_table_df['Description'] = selected_table_df['Description'].str.replace('-', ' ', regex=False)

                for col in selected_table_df.columns[2:]:
                    selected_table_df[col] = selected_table_df[col].astype(str).str.replace(',', '')
                    selected_table_df[col] = pd.to_numeric(selected_table_df[col], errors='coerce')

                note_mask = selected_table_df['Note'].astype(str).str.strip() == "Note"
                if note_mask.any():
                    first_note_idx = note_mask.idxmax()
                    selected_table_df = selected_table_df.loc[first_note_idx + 1:].reset_index(drop=True)

                # Merge "Year/Period" rows
                table_one = selected_table_df.copy()
                mask = table_one['Description'].str.strip().str.lower() == "year/period"
                for idx in table_one[mask].index:
                    if idx > 0:
                        table_one.at[idx - 1, 'Description'] += ' ' + table_one.at[idx, 'Description']
                        for col in ['Note', 'Year', 'Previous year']:
                            prev_val = table_one.at[idx - 1, col]
                            curr_val = table_one.at[idx, col]
                            if (pd.isna(prev_val) or prev_val in ['', None]) and pd.notna(curr_val) and curr_val != '':
                                table_one.at[idx - 1, col] = curr_val

                selected_table_df = table_one[~mask].reset_index(drop=True)

            else:
                for col in selected_table_df.columns[1:]:
                    selected_table_df[col] = selected_table_df[col].astype(str).str.replace(',', '')
                    selected_table_df[col] = pd.to_numeric(selected_table_df[col], errors='coerce')

        # Merge tables under same header
        if header_found in tables_dict:
            selected_table_df.insert(0, 'index', range(1, len(selected_table_df) + 1))
            tables_dict[header_found] = pd.concat(
                [tables_dict[header_found], selected_table_df], ignore_index=True
            )
        else:
            selected_table_df.insert(0, 'index', range(1, len(selected_table_df) + 1))
            tables_dict[header_found] = selected_table_df

        # Stop after first Notes table
        if header_found == 'Notes to the Financial Statements':
            break
    

    


    return tables_dict



"""Preprocess the tables only matchs total 4 columns"""

def preprocess_first_two_tables(tables_dict):
    processed_tables = {}

    for i, (table_name, table_df) in enumerate(tables_dict.items()):
        if i >= 2:
            break

        if table_df.shape[1] == 4:
            # Remove rows where 'Note' is empty
            table_one = table_df[table_df['Note'].astype(str).str.strip() != ''].reset_index(drop=True)

            # Sort based on numeric 'Note'
            # table_one['sort'] = pd.to_numeric(table_one['Note'], errors='coerce')
            # table_one = table_one.sort_values(by='sort', ascending=True).reset_index(drop=True)
            # table_one.drop(columns='sort', inplace=True)

            processed_tables[table_name] = table_one

    return processed_tables





"""Validated the Header and index present in pdf"""

def statements_index_checker(pdf_path, table):
    doc = pdf_open(pdf_path)
    table_results = []

    for _, row in table.iterrows():
        note = str(row['Note']).strip()
        note_value = f"{note}."

        description = str(row['Description']).lower().replace("’", "'") \
            .replace("-", " ").replace(":", " ").strip()
        # description = re.sub(r"\s+", " ", description).rstrip('s')
        description_cont = f"{description} (continued)"

        value_inline = f"{note_value} {description}"
        value_inline_cont = f"{note_value} {description_cont}"

        Year=row['Year']
        Previous_year=row['Previous year']

        matched_entries = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            lines = page.get_text().splitlines()

            # Normalize and clean lines
            cleaned_lines = [
                re.sub(r"\s+", " ", line.lower().replace("’", "'")
                       .replace("-", " ").replace(":", " ").strip()) #.rstrip('s')
                for line in lines
            ]

            for i, line in enumerate(cleaned_lines):
                if line == value_inline or line == value_inline_cont:
                    matched_entries.append({
                        'Page_NO': page_num - 1,
                        'Header_value': line,
                        'year'        : Year,
                        'Previous year':Previous_year,
                        'Header_Found': 'Present'
                    })
                elif line == note_value and i + 1 < len(cleaned_lines):
                    next_line = cleaned_lines[i + 1]
                    if next_line in description or next_line in description_cont:
                        matched_entries.append({
                            'Page_NO': page_num - 1,
                            'Header_value': f"{line} {next_line}",
                            'year'        : Year,
                            'Previous year':Previous_year,
                            'Header_Found': 'Present'
                        })

        # If nothing matched, append Not Present
        if not matched_entries:
            table_results.append({
                'Page_NO': 'Not Present',
                'Header_value': value_inline,
                'year'        : Year,
                'Previous year':Previous_year,
                'Header_Found': 'Not Present'
            })
        else:
            table_results.extend(matched_entries)

    return pd.DataFrame(table_results)





def statement_index_and_header_validation(pdf_path,tables_dict):
    two_tables = preprocess_first_two_tables(tables_dict)
    total_validated =[]
    for table_name, table in two_tables.items():
        result = statements_index_checker(pdf_path, table)
        # result_df['Page_NO'].astype(int)
        result=result.drop_duplicates()
        result = result[result['Header_value'].astype(str).str.strip() != '']
        total_validated.append(result)
        
    full_validation=pd.concat(total_validated, axis=0, ignore_index=True)

    return full_validation