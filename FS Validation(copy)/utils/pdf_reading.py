import fitz  # PyMuPDF

def pdf_open(path):
    try:
        doc = fitz.open(path)
        return doc
    except Exception as e:
        print(f"Failed to open PDF: {e}")
        return None
