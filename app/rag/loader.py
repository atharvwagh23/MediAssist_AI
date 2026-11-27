"""
Unified document loader.

Reads PDF, DOCX and TXT files and returns the same simple format for all of
them: a list of "pages", where each page is a dictionary with the text, the
source file name and the page number.
"""

from pathlib import Path  # file paths that work on Windows

from docx import Document  # reads .docx files (python-docx)
from pypdf import PdfReader  # reads .pdf files


def load_pdf(path: Path) -> list[dict]:  # one entry per PDF page
    reader = PdfReader(str(path))  # open the PDF
    pages = []  # result list
    for number, page in enumerate(reader.pages, start=1):  # page numbers start at 1
        text = (page.extract_text() or "").strip()  # empty string if the page has no text
        if text:  # skip blank pages
            pages.append({"text": text, "source": path.name, "page": number})
    return pages


def load_docx(path: Path) -> list[dict]:  # a Word file is treated as one page
    document = Document(str(path))  # open the Word file
    parts = [p.text for p in document.paragraphs if p.text.strip()]  # normal paragraphs
    for table in document.tables:  # tables are stored separately in Word
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))  # one table row becomes one line
    text = "\n".join(parts).strip()
    return [{"text": text, "source": path.name, "page": 1}] if text else []


def load_txt(path: Path) -> list[dict]:  # a text file is treated as one page
    text = path.read_text(encoding="utf-8", errors="replace").strip()  # replace bad characters
    return [{"text": text, "source": path.name, "page": 1}] if text else []


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt}  # extension -> function


def load_document(path) -> list[dict]:
    """Load any supported document into the common page format."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())  # pick the function by extension
    if loader is None:  # unsupported type
        raise ValueError(f"Unsupported file type: {path.suffix}")
    pages = loader(path)
    if not pages:  # e.g. a scanned PDF with no text layer
        raise ValueError(f"No readable text found in {path.name}")
    return pages