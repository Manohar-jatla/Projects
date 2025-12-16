import fitz  # PyMuPDF
from io import BytesIO

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
