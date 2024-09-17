import pdfminer.high_level

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        extracted_text = pdfminer.high_level.extract_text(f)

    return extracted_text

if name == "main":
    pdf_path = "C:\Users\USER\Desktop.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)