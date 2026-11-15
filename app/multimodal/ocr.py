"""
OCR: reads ALL the text lines from an image with EasyOCR, in reading order.

Nothing is thrown away: even low-confidence lines are kept (the confidence is
reported so the user can be warned).
"""
import re  # pattern fixes for common OCR mistakes
import threading  # lets only one OCR run use the model at a time

import numpy as np  # EasyOCR takes the picture as a number array

from app import config

_reader = None  # the EasyOCR model, created once because loading is slow
_lock = threading.Lock()  # two requests must not use the model at the same moment


def get_reader():  # load EasyOCR the first time it is needed
    """Return the shared EasyOCR reader (downloads its model files on first use)."""
    global _reader
    if _reader is None:
        import easyocr  # imported here so the app starts fast and tests run without it
        _reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=False)  # CPU only: works on every laptop
    return _reader

def fix_common_ocr_errors(line: str) -> str:
    """Fix a few mistakes EasyOCR makes again and again (found in real test runs).

    The slash in units is read as a letter: gldL -> g/dL, mgldL -> mg/dL, luL -> /uL.
    The letter O inside a patient ID is read for zero: PO0123 -> P00123.
    """
    line = re.sub(r"\b(m?g)l(dL)\b", r"\1/\2", line)  # g/dL and mg/dL
    line = re.sub(r"(?<![A-Za-z])[l/]uL\b", "/uL", line)  # /uL
    line = re.sub(r"\bP[O0](\d{3,})\b", r"P0\1", line)  # patient ID
    return line

def group_into_lines(boxes: list) -> list[dict]:
    """boxes = EasyOCR output [(corners, text, confidence), ...] in any order.

    Returns [{"text", "confidence"}], one entry per printed line, top to bottom.
    Words that share the same height are joined left to right.
    """
    items = []
    for corners, text, confidence in boxes:  # measure every detected box
        text = str(text).strip()
        if not text:  # skip empty boxes only
            continue
        xs = [point[0] for point in corners]  # left/right edges
        ys = [point[1] for point in corners]  # top/bottom edges
        items.append({"text": text, "conf": float(confidence), "x": min(xs),
                      "top": min(ys), "bottom": max(ys)})

    items.sort(key=lambda item: item["top"] + item["bottom"])  # top to bottom (by vertical centre)
    rows: list[list[dict]] = []  # each row = the words of one line
    for item in items:
        if rows:
            row = rows[-1]
            row_top = min(i["top"] for i in row)  # vertical span of the current line
            row_bottom = max(i["bottom"] for i in row)
            overlap = min(item["bottom"], row_bottom) - max(item["top"], row_top)  # shared height
            smaller = max(min(item["bottom"] - item["top"], row_bottom - row_top), 1)
            if overlap >= 0.5 * smaller:  # mostly side by side = same line (a small "-" still fits)
                row.append(item)
                continue
        rows.append([item])  # otherwise start a new line

    lines = []
    for row in rows:
        row.sort(key=lambda item: item["x"])  # left to right inside the line
        lines.append({"text": fix_common_ocr_errors(" ".join(i["text"] for i in row)),
                      "confidence": round(sum(i["conf"] for i in row) / len(row), 3)})
    return lines


def extract_text(image) -> dict:
    """Read all text from a PIL image.

    Returns {"text": all lines joined by newlines, "lines": [...], "line_count": n,
             "confidence": average 0-1}.
    """
    array = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])  # RGB -> BGR for EasyOCR
    with _lock:  # one run at a time
        boxes = get_reader().readtext(array, detail=1, paragraph=False)  # [(corners, text, confidence)]
    lines = group_into_lines(boxes)
    confidence = round(sum(line["confidence"] for line in lines) / len(lines), 3) if lines else 0.0
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines,
            "line_count": len(lines), "confidence": confidence}