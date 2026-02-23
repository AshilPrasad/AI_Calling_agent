import pandas as pd
from dateutil import parser
import fitz  # PyMuPDF
from typing import Tuple, Dict, Any
from utils.pdf_reading import pdf_open



""" 
Validate presence of signatories, registration numbers, and audit date 
from the last page that contains the audit report header 

"""



def signatory_and_date_validation(pdf_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    # Constants
    year_end = "For the year ended December 31, 2023"
    expected_date = parser.parse(year_end, fuzzy=True).date()
    report_header = "Independent Auditor's Report (continued)"
    
    # Expected signatories and firm info
    signatory_info = {
        'Vijay Anand': '[Reg. No. 654]',
        'Hisham': '[Reg. No. 5397]',
        'Jay Krishnan': '[Reg. No. 5717]',
        'Manoj Kumar': '[Reg. No. 5713]',
        'Reg number': '[Firm Reg. No. LC0075-01]',
        'Date': expected_date
    }

    signatory_data = []
    status_data: Dict[str, Any] = {}
    pages_with_header = []

    try:
        doc = pdf_open(pdf_path)

        # Step 1: Find last page with report header
        for page_num in range(2, len(doc)):
            page_text = doc.load_page(page_num).get_text()
            lines = page_text.splitlines()[:5]
            for line in lines:
                if line.strip().lower() == report_header.strip().lower():
                    pages_with_header.append(page_num)
                    break

        if not pages_with_header:
            status_data['Date Status'] = "Report Header Not Found"
            status_data['Firm Reg number Status'] = "Report Header Not Found"
            for name, reg in signatory_info.items():
                if name in ['Date', 'Reg number']:
                    continue
                signatory_data.append({
                    'Name': name,
                    'Reg. NO': reg,
                    'Name Status': 'Not Present',
                    'Reg No Status': 'Not Present'
                })
            return pd.DataFrame(signatory_data), pd.DataFrame([status_data])

        # Step 2: Extract text from last report page
        last_report_page = doc.load_page(pages_with_header[-1])
        last_lines = last_report_page.get_text().splitlines()[-10:]

        # Step 3: Extract audit date
        found_date = None
        for line in last_lines:
            try:
                found_date = parser.parse(line, fuzzy=True).date()
                break
            except:
                continue

        # Step 4: Validate audit date
        if found_date:
            status_data['Date Status'] = (
                "Date Condition True" if found_date >= expected_date else "Date Condition False"
            )
        else:
            status_data['Date Status'] = "Date Not Found"

        # Step 5: Validate firm registration number
        firm_reg = signatory_info['Reg number']
        status_data['Firm Reg number Status'] = (
            "Present" if any(firm_reg in line for line in last_lines) else "Not Present"
        )

        # Step 6: Validate individual signatories
        for name, reg in signatory_info.items():
            if name in ['Date', 'Reg number']:
                continue
            name_found = any(name.lower() in line.lower() for line in last_lines)
            reg_found = any(reg in line for line in last_lines)

            signatory_data.append({
                'Name': name,
                'Reg. NO': reg,
                'Name Status': 'Present' if name_found else 'Not Present',
                'Reg No Status': 'Present' if reg_found else 'Not Present'
            })

        return pd.DataFrame(signatory_data), pd.DataFrame([status_data])

    except Exception as e:
        raise RuntimeError(f"Error during validation: {str(e)}")


# signatory_df, status_df = signatory_check(pdf_path, report_header,signatory)
# signatory_df