import pdfplumber
from docx import Document

def extract_text_from_file(file):
    """
    Extract text from PDF, DOCX, or TXT uploaded via Streamlit.
    """

    filename = file.name.lower()

    # ---- PDF ----
    if filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                return text
        except Exception as e:
            return f"Error reading PDF: {e}"

    # ---- DOCX ----
    elif filename.endswith(".docx"):
        try:
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Error reading DOCX: {e}"

    # ---- TXT ----
    elif filename.endswith(".txt"):
        try:
            return file.read().decode("utf-8", errors="ignore")
        except:
            return file.read().decode("latin-1", errors="ignore")

    # Unknown format
    return "Unsupported file format."