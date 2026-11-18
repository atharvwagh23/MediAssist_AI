"""
OCR: reads ALL the text lines from an image with EasyOCR, in reading order.

Accuracy steps (in order):
  1. Clean-up: grey, upscale, denoise, local contrast (CLAHE).
  2. First pass with the slower but more accurate beam-search decoder.
  3. If confidence is low, a second pass on a black-and-white version. The better pass wins.
  4. Fix known unit/ID mistakes and near-miss medicine names.

Nothing is thrown away: even low-confidence lines are kept (the confidence is
reported so the user can be warned).
"""
import difflib  # close-match search for medicine names
import re  # pattern fixes for common OCR mistakes
import threading  # lets only one OCR run use the model at a time

import cv2  # installed together with EasyOCR
import numpy as np  # EasyOCR takes the picture as a number array

from app import config

_reader = None  # the EasyOCR model, created once because loading is slow
_lock = threading.Lock()  # two requests must not use the model at the same moment

GOOD_ENOUGH = 0.85  # if the first pass is this confident, skip the second pass
TARGET_SIDE = 1800  # small images are enlarged to about this size (letters need pixels)

# Common medicines. A word that is very close to one of these is corrected to it.
MEDICINES = [
    "paracetamol", "acetaminophen", "ibuprofen", "diclofenac", "aspirin", "naproxen", "tramadol",
    "amoxicillin", "amoxycillin", "augmentin", "azithromycin", "cefixime", "cefuroxime", "ceftriaxone",
    "ciprofloxacin", "levofloxacin", "doxycycline", "metronidazole", "clindamycin", "cotrimoxazole",
    "metformin", "glimepiride", "gliclazide", "sitagliptin", "insulin", "atorvastatin", "rosuvastatin",
    "amlodipine", "losartan", "telmisartan", "olmesartan", "metoprolol", "atenolol", "bisoprolol",
    "clopidogrel", "warfarin", "furosemide", "spironolactone", "hydrochlorothiazide", "enalapril",
    "omeprazole", "pantoprazole", "rabeprazole", "esomeprazole", "ranitidine", "famotidine",
    "domperidone", "ondansetron", "cetirizine", "levocetirizine", "montelukast", "fexofenadine",
    "salbutamol", "prednisolone", "dexamethasone", "levothyroxine", "thyroxine", "folic", "vitamin",
    "calcium", "ferrous", "multivitamin", "loperamide", "lactulose", "gabapentin", "pregabalin",
    "sertraline", "escitalopram", "alprazolam", "clonazepam", "diazepam",
]


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
    line = re.sub(r"\b(m?g)[1I|]\s?(dL)\b", r"\1/\2", line)  # mg1dL, mg|dL
    line = re.sub(r"(?<![A-Za-z])[l/]uL\b", "/uL", line)  # /uL
    line = re.sub(r"\bP[O0](\d{3,})\b", r"P0\1", line)  # patient ID
    line = re.sub(r"\b(\d+)\s?m9\b", r"\1 mg", line)  # 500m9 -> 500 mg
    line = re.sub(r"\b(\d+)\s?rng\b", r"\1 mg", line)  # 500rng -> 500 mg
    return line


def correct_medicines(line: str) -> str:
    """Replace a word that is almost a known medicine name (at least 85% alike) with the real name."""
    def fix(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if len(lower) < 5 or lower in MEDICINES:
            return word
        close = difflib.get_close_matches(lower, MEDICINES, n=1, cutoff=0.85)
        if not close:
            return word
        return close[0].upper() if word.isupper() else close[0].capitalize()
    return re.sub(r"[A-Za-z]{5,}", fix, line)


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
        text = " ".join(i["text"] for i in row)
        lines.append({"text": correct_medicines(fix_common_ocr_errors(text)),
                      "confidence": round(sum(i["conf"] for i in row) / len(row), 3)})
    return lines


def enhance(image) -> np.ndarray:
    """Grey, enlarge small images, remove noise, even out lighting. Returns a grey array."""
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    longest = max(gray.shape)
    if longest < TARGET_SIDE:  # tiny letters read badly: enlarge (at most 3 times)
        scale = min(TARGET_SIDE / longest, 3.0)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # local contrast: shadows, uneven light
    return clahe.apply(gray)


def black_and_white(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold: helps faded print, phone photos with shadows, stamps and watermarks."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def read_once(array: np.ndarray) -> list[dict]:
    """One OCR pass with accuracy-focused settings."""
    with _lock:  # one run at a time
        boxes = get_reader().readtext(
            array, detail=1, paragraph=False,
            decoder="beamsearch", beamWidth=5,  # slower than greedy, but fewer wrong letters
            text_threshold=0.6, low_text=0.3, link_threshold=0.4,  # keep faint text
            contrast_ths=0.3, adjust_contrast=0.7,  # re-try low-contrast boxes with more contrast
            width_ths=0.5, mag_ratio=1.0)
    return group_into_lines(boxes)


def pass_score(lines: list[dict]) -> float:
    """Characters read, weighted by confidence. Higher = a better pass."""
    return sum(line["confidence"] * len(line["text"]) for line in lines)


def average_confidence(lines: list[dict]) -> float:
    return round(sum(line["confidence"] for line in lines) / len(lines), 3) if lines else 0.0


def extract_text(image) -> dict:
    """Read all text from a PIL image.

    Returns {"text": all lines joined by newlines, "lines": [...], "line_count": n,
             "confidence": average 0-1}.
    """
    gray = enhance(image)
    lines = read_once(gray)  # pass 1: cleaned grey image
    if average_confidence(lines) < GOOD_ENOUGH:  # pass 2 only when pass 1 is unsure
        second = read_once(black_and_white(gray))
        if pass_score(second) > pass_score(lines):
            lines = second
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines,
            "line_count": len(lines), "confidence": average_confidence(lines)}