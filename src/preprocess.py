import re
import pdfplumber


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/non-letters, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text(file) -> str:
    """Read text from an uploaded PDF or TXT (Streamlit UploadedFile or file object)."""
    name = getattr(file, "name", "").lower()
    if hasattr(file, "seek"):
        file.seek(0)  # Streamlit reruns the script; always read from the start
    if name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    return file.read().decode("utf-8", errors="ignore")


def is_low_text(raw: str, min_words: int = 50) -> bool:
    """True if almost no text was extracted (e.g. scanned/image PDF or odd layout)."""
    return len(raw.split()) < min_words
