"""
File guardrail: decides whether an uploaded file is safe to accept.

Checks: file count, name, allowed extension, empty file, size limit, REAL file
type (first bytes, not just the extension), hidden executables, Word macros,
zip bombs, risky PDF features, broken images. A rejected file is never saved.

This is a practical safety check, not a full antivirus. ClamAV can be added
later for deeper scanning.
"""

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app import config
from app.guardrails.input_guard import find_injection

PDF_RISKY = (b"/JavaScript", b"/Launch", b"/EmbeddedFile", b"/OpenAction", b"/RichMedia")
EXECUTABLE_SIGNATURES = (b"MZ", b"\x7fELF", b"#!/", b"<?php", b"<script")
MAX_UNZIPPED_BYTES = 300 * 1024 * 1024  # zip-bomb limit for .docx
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class FileCheck:
    ok: bool
    status: int = 200      # HTTP status to use when rejected
    message: str = ""      # text to show the user
    safe_name: str = ""
    ext: str = ""
    kind: str = ""         # "document" or "image"


def _reject(status: int, message: str) -> FileCheck:
    return FileCheck(ok=False, status=status, message=message)


def _check_docx(content: bytes) -> str | None:
    if not zipfile.is_zipfile(io.BytesIO(content)):
        return "This is not a real Word (.docx) file."
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if "word/document.xml" not in names or "[Content_Types].xml" not in names:
            return "This is not a real Word (.docx) file."
        if any(n.lower().endswith("vbaproject.bin") for n in names):
            return "Files with macros are not allowed."
        if any(n.startswith(("/", "..")) or ".." in Path(n).parts for n in names):
            return "The file contains unsafe paths."
        if sum(i.file_size for i in archive.infolist()) > MAX_UNZIPPED_BYTES:
            return "The file expands to an unsafe size."
    return None


def _check_image(content: bytes, ext: str) -> str | None:
    if ext == "png" and not content.startswith(PNG_SIGNATURE):
        return "This is not a real PNG image."
    if ext in ("jpg", "jpeg") and not content.startswith(b"\xff\xd8\xff"):
        return "This is not a real JPG image."
    try:
        from PIL import Image  # Pillow: confirm the image can really be opened
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except ImportError:
        pass
    except Exception:
        return "The image is damaged or not a valid image."
    return None


def check_file(filename: str | None, content: bytes, file_count: int = 1) -> FileCheck:
    """Validate one uploaded file. Returns FileCheck(ok=True, ...) or a rejection."""
    if file_count > config.MAX_FILES_PER_UPLOAD:
        return _reject(400, f"Upload {config.MAX_FILES_PER_UPLOAD} file at a time.")

    original = Path((filename or "").replace("\\", "/")).name  # drop any folder part
    if not original:
        return _reject(400, "The file has no name.")
    ext = Path(original).suffix.lower().lstrip(".")
    allowed = config.ALLOWED_DOC_TYPES + config.ALLOWED_IMAGE_TYPES
    if ext not in allowed:
        return _reject(400, f"Unsupported file type '.{ext}'. Allowed: {', '.join(allowed)}.")
    if not content:
        return _reject(400, "The file is empty.")
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return _reject(413, f"The file is larger than {config.MAX_UPLOAD_MB} MB.")
    if content.startswith(EXECUTABLE_SIGNATURES):
        return _reject(400, "Executable or script content is not allowed.")

    problem = None
    if ext == "pdf":
        if not content.startswith(b"%PDF-"):
            problem = "This is not a real PDF file."
        elif any(token in content for token in PDF_RISKY):
            problem = "The PDF contains scripts or embedded files, which are not allowed."
    elif ext == "docx":
        problem = _check_docx(content)
    elif ext == "txt":
        if b"\x00" in content:
            problem = "This is not a plain text file."
        elif content.startswith((b"%PDF-", b"PK\x03\x04")):
            problem = "This file is not plain text."
    elif ext in config.ALLOWED_IMAGE_TYPES:
        problem = _check_image(content, ext)
    if problem:
        return _reject(400, problem)

    safe_name = re.sub(r"[^\w.\- ]", "_", original)  # letters, digits, . - and spaces only
    kind = "image" if ext in config.ALLOWED_IMAGE_TYPES else "document"
    return FileCheck(ok=True, safe_name=safe_name, ext=ext, kind=kind)


def check_document_text(text: str) -> FileCheck:
    """Scan the TEXT inside a document for prompt-injection phrases (run after loading)."""
    if find_injection(text):
        return _reject(422, "The document contains instructions that try to change the assistant's "
                            "behaviour, so it was not accepted.")
    return FileCheck(ok=True)
