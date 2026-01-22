import io
import unicodedata
from docx import Document
import pandas as pd
import pdfplumber


def read_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="ignore")


def read_docx_bytes(b: bytes) -> str:
    f = io.BytesIO(b)
    doc = Document(f)
    return "\n".join([p.text for p in doc.paragraphs])


def read_csv_bytes(b: bytes) -> str:
    f = io.BytesIO(b)

    try:
        df = pd.read_csv(f)
    except Exception:
        f.seek(0)
        df = pd.read_csv(f, sep=";")
    return "\n".join(df.astype(str).fillna("").agg("".join, axis=1).tolist())


def read_pdf_bytes(b: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts)


def clean_text(s: str) -> str:
    # Normalize để gom các biến thể unicode về dạng “chuẩn”
    s = unicodedata.normalize("NFKC", s)

    # Thay các ký tự hay gây rắc rối
    s = s.replace("×", "x")  # hoặc " " nếu bạn muốn bỏ hẳn

    # Bỏ NULL và các control characters (trừ \n, \t)
    s = s.replace("\x00", "")
    s = "".join(ch for ch in s if (ch in "\n\t") or (unicodedata.category(ch)[0] != "C"))

    return s


def read_upload_file(uploaded) -> str:
    b = uploaded.getvalue()
    name = uploaded.name.lower()

    if name.endswith(".docx"):
        text = read_docx_bytes(b)
    elif name.endswith(".csv"):
        text = read_csv_bytes(b)
    elif name.endswith(".pdf"):
        text = read_pdf_bytes(b)
    else:
        text = read_txt_bytes(b)

    return clean_text(text)
