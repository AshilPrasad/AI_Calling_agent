
import pandas as pd
from utils.pdf_reading import pdf_open

expected_values = {
    "Company Name": "FAKHRUDDIN PROPERTIES DEVELOPMENT L.L.C",
    "Emirate": "Dubai - United Arab Emirates",
    "Year End": "For the year ended December 31, 2023"
}


def detect_header_blocks(page, max_y=115):
   
    blocks = page.get_text("dict")["blocks"]
    header_lines = []
    
    for block in blocks:
        if 'lines' not in block:
            continue
        if block['bbox'][1] < max_y:  # Check if block is in header region (y-coordinate)
            for line in block['lines']:
                line_text = " ".join(span['text'] for span in line['spans']).strip()
                if line_text:
                    header_lines.append(line_text)
   
    return header_lines


def validate_header_structure(header_lines, expected_values):
 
    missing = []
    
    for key, expected_text in expected_values.items():
        found = False
        for line in header_lines:
            if key == "Year End":
                # Check if "Year End" text is contained in the line
                if line.lower() == expected_text.lower():
                    found = True
                    break
            else:
                # Check for exact match (case insensitive) for other fields
                if expected_text.lower() == line.lower():
                    found = True
                    break
        
        if not found:
            missing.append(key)
    
    return missing

def header_validations(pdf_path):
    
    results = []

    doc = pdf_open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        if page_num == 0:  # Cover page - check full text
            page_text = page.get_text().lower()
            missing = [
                field for field, val in expected_values.items() 
                if val.lower() not in page_text
            ]
        else:  # Other pages - check header structure
            header_lines = detect_header_blocks(page)
            missing = validate_header_structure(header_lines, expected_values)
        
        if missing:
            results.append({
                "Page Number": page_num - 1,
                "Missing Fields": ", ".join(missing),
            })

    return pd.DataFrame(results)

# missing_valus = check_dynamic_headers(pdf_path,expected_values)
# missing_valus
