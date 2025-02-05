from PyPDF2 import PdfReader

def extract_text_from_pdf():
    pdf_path = "manual.pdf"
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text()
    return text

if __name__ == "__main__":
    text = extract_text_from_pdf()
    with open("manual_text.txt", "w") as text_file:
        text_file.write(text)
