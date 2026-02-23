import pandas as pd
from utils.pdf_reading import pdf_open



def directors_report_validation(pdf_path):
    
    doc = pdf_open(pdf_path)
    data = []
    report_title = "DIRECTOR'S REPORT"

    for page_num in range(2, len(doc)):
        page = doc.load_page(page_num)
        lines = page.get_text().splitlines()[:10]

        for line in lines:
            if line.strip().lower() == report_title.strip().lower():
                data.append({
                    'Page number': page_num - 1,
                    'Heading': line.strip()
                })
                break

    if not data:
        data.append({
            'Page number': None,
            'Heading': "Director's heading not  present"
        })

    return pd.DataFrame(data)

# matched_pages = find_directors_report_pages(pdf_path, report_title)
# matched_pages
