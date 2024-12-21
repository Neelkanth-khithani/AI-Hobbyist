import fitz  

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)  
    text = ""
    for page in doc:
        text += page.get_text()  
    return text