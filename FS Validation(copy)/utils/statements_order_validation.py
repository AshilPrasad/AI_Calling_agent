import pandas as pd
from utils.pdf_reading import pdf_open



def statement_order_validation(pdf_path, main_statements_order, all_possible_headers):
    doc = pdf_open(pdf_path)
    order = []

    for page_num in range(2, len(doc)):  # Start from page 3 (index 2)
        page = doc.load_page(page_num)
        lines = page.get_text().splitlines()[:10]  # Only check first 7 lines

        matched_header = None
        for line in lines:
            line = line.strip()
            for header in all_possible_headers:
                if line == header:  # Exact match
                    matched_header = header
                    break
            if matched_header:
                break

        order.append({
            'Page_NO': page_num + 1,
            'Header_Found': matched_header,
            'Main_Header': matched_header.replace(' (continued)', '') if matched_header else matched_header
        })

    df = pd.DataFrame(order)
 

    current_max_index = -1
    validations = []

    for _, row in df.iterrows():
        main_header = row['Main_Header']
        
        if pd.isna(main_header):
            validations.append("Invalid")
            continue

        try:
            index_in_order = main_statements_order.index(main_header)
        except ValueError:
            validations.append("Invalid")
            continue

        if index_in_order == current_max_index or index_in_order == current_max_index + 1:
            current_max_index = max(current_max_index, index_in_order)
            validations.append("Valid")
        elif index_in_order < current_max_index:
            validations.append("Invalid")  # Repeated earlier header → Invalid
        else:
            validations.append("Invalid")  # Skipped expected headers → Invalid

    df['Validation'] = validations
    # result=df.drop(columns=["Main_Header"])
    return df


# result = statements_header(pdf_path, main_statements_order, all_possible_headers)
# result