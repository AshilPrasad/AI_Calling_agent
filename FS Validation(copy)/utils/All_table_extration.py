
import re
import camelot
import numpy as np
import pandas as pd
from utils.pdf_reading import pdf_open




"""Clean the Each Extracted Tables"""

def clean_and_transform_table(df):
    if df.shape[1] >= 2:
        df = df.drop(df.index[-1])  # Drop last row (often footnote or empty)

        # Clean first column: blank/whitespace → NaN
        df.iloc[:, 0] = df.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True)

        # Clean numeric columns
        for col in df.columns[1:]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.replace(r'\((.*?)\)', r'-\1', regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop empty columns and rows
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        # df = df.fillna(0)

    return df




"""Extract each table from the pdf"""

def all_table_extraction(pdf_path, extracted_table_validation):
    pdf_tables = {}
    min_page = int(pd.to_numeric(extracted_table_validation['Page_NO'], errors='coerce').min())
    max_page = len(pdf_open(pdf_path))

    for page_num in range(min_page, max_page):
        actual_page = page_num + 2  # Adjust for index shift if needed

        if actual_page <= max_page:
            try:
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=str(actual_page),
                    flavor='stream',
                    strip_text='\n',
                    table_areas=['50,800,900,50'],
                    row_tol=8,
                    layout_kwargs={
                        'detect_vertical': True,
                        'line_overlap': 0.1,
                        'char_margin': 3,
                        'line_margin': 0.1,
                        'word_margin': 0.2,
                        'boxes_flow': 0.0
                    }
                )

                if tables and len(tables) > 0:
                    last_df = tables[-1].df
                    cleaned_df = clean_and_transform_table(last_df)

                    if cleaned_df.shape[1] >= 2:
                        cleaned_df.attrs["page_num"] = actual_page - 2
                        pdf_tables[actual_page - 2] = cleaned_df

            except Exception as e:
                print(f"⚠️ Error on page {actual_page}: {e}")

    return pdf_tables



"""Extract the subtables from the each tables"""

def extract_subtables_from_main_table(pdf_tables):
    final_tables_dict = {}
    pattern = re.compile(r'^(\d+(?:\.\d+)*)(?:\.)?\s+(.+)$')

    for page, table in pdf_tables.items():
        df = table.copy().reset_index(drop=True)
        first_col = df.iloc[:, 0].astype(str).str.strip()

        # Find heading rows using the pattern like '5.1 Title'
        matches = [
            (idx, match.group(0))
            for idx, val in enumerate(first_col)
            if (match := pattern.match(val))
        ]

        if not matches:
            continue

        for i, (match_idx, heading) in enumerate(matches):
            start_idx = match_idx
            end_idx = matches[i + 1][0] if i + 1 < len(matches) else len(df)
            sub_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

            # Smarter header detection: if first row is mostly text, treat as header
            first_row = sub_df.iloc[0]
            if all(isinstance(val, str) or pd.isna(val) for val in first_row):
                sub_df.columns = sub_df.iloc[0]
                sub_df = sub_df[1:].reset_index(drop=True)
            else:
                sub_df.columns = [f"Column_{j}" for j in range(sub_df.shape[1])]

            # Clean empty rows and columns
            sub_df = sub_df.dropna(axis=0, how='all')
            sub_df = sub_df.dropna(axis=1, how='all')

            # Skip invalid subtables
            if sub_df.shape[1] <= 1:
                continue

            # ✅ Rename logic: Only if first value in col is non-NaN AND first column is NaN
            for col_idx in range(1, sub_df.shape[1]):  # skip first column
                for row_idx in range(len(sub_df)):
                    col_val = sub_df.iloc[row_idx, col_idx]
                    first_col_val = sub_df.iloc[row_idx, 0]

                    if pd.notna(col_val):
                        if pd.isna(first_col_val):
                            # Rename the column and replace the value with NaN
                            new_columns = list(sub_df.columns)
                            new_columns[col_idx] = str(col_val)
                            sub_df.columns = new_columns
                            sub_df.iat[row_idx, col_idx] = np.nan
                        # Stop after first non-NaN value, regardless of header update
                        break
            sub_df.columns = [sub_df.columns[0]] + list(map(str, range(1, len(sub_df.columns))))
            final_tables_dict[heading] = sub_df

    return final_tables_dict


