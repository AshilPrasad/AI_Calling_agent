import re
import pandas as pd
from utils import extract_subtables_from_main_table
from utils import all_table_extraction
from utils import statement_index_and_header_validation



def cross_checking(pdf_path: str, tables_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:

    extracted_table_validation = statement_index_and_header_validation(pdf_path, tables_dict)
    each_page_tables = all_table_extraction(pdf_path, extracted_table_validation)
    final_subtables = extract_subtables_from_main_table(each_page_tables)

    extracted_table_validation['Header_Found'] = 'Present'
    cross_check_results = []

    for idx, header_row in extracted_table_validation.iterrows():
        header_raw = header_row.get('Header_value')
        if pd.isna(header_raw):
            continue

        header_name = str(header_raw).strip().lower()
        year_val = header_row.get('year')
        prev_year_val = header_row.get('Previous year')

        for table_name, table in final_subtables.items():
            cleaned_name = re.sub(r'\s+', ' ', table_name).strip().lower()

            if header_name == cleaned_name:
                year_result = 'NaN' if pd.isna(year_val) else 'Not Match'
                prev_year_result = 'NaN' if pd.isna(prev_year_val) else 'Not Match'

                for col in table.columns[1:]:
                    col_series = table[col]
                    if not pd.isna(year_val) and col_series.eq(year_val).any():
                        year_result = 'Match'
                    if not pd.isna(prev_year_val) and col_series.eq(prev_year_val).any():
                        prev_year_result = 'Match'

                cross_check_results.append({
                    'Header': table_name,
                    'Year': year_result,
                    'Previous Year': prev_year_result
                })
                break

    return pd.DataFrame(cross_check_results)
