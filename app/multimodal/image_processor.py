"""
Image processor: opens an uploaded image, prepares it, runs OCR, describes it
with BLIP (vision) and guesses what kind of document it is.

process_image(path) is the one function other files call.
"""

import logging
import re
import threading
import time
from pathlib import Path

from PIL import Image, ImageOps  # Pillow: open, rotate, resize, contrast

from app import config
from app.multimodal import ocr

logger = logging.getLogger("mediassist.multimodal")

MIN_TEXT_WORDS = 5  # fewer words than this = "a picture without readable text"

# Words that hint at the kind of document (each pattern counts once).
PRESCRIPTION_HINTS = [
    r"\brx\b", r"\b(tab|tablet|tablets)\b", r"\b(cap|capsule|capsules)\b", r"\b(syp|syrup)\b",
    r"\binj\b", r"\b(od|bd|tds|tid|qid|sos|hs)\b", r"\bprescription\b", r"\b\d+\s?mg\b",
    r"\b(daily|twice)\b", r"(after|before) food", r"\bdr\.?\b",
]
LAB_HINTS = [
    r"h(a)?emoglobin", r"\b(wbc|rbc)\b", r"platelet", r"\breference\b|\brange\b", r"\bresults?\b",
    r"glucose|creatinine|cholesterol|urea|bilirubin", r"mg/dl|g/dl|/ul|mmol",
    r"\bspecimen|\bpatholog|\blaborator|\blab\b", r"\bunit\b",
]

_vision = None  # (processor, model) for BLIP, loaded once
_vision_lock = threading.Lock()


def load_image(path) -> Image.Image:
    """Open an image safely: fix phone rotation, convert to RGB."""
    try:
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")  # exif_transpose = rotate like the phone did
    except Exception as exc:  # damaged file, decompression bomb, wrong format...
        raise ValueError("The image could not be opened.") from exc


def prepare_for_ocr(image: Image.Image) -> Image.Image:
    """Resize (small -> bigger, huge -> smaller) and boost contrast so OCR reads better."""
    longest = max(image.size)
    if longest > config.OCR_MAX_SIDE:  # too big = slow
        scale = config.OCR_MAX_SIDE / longest
    elif longest < 1000:  # too small = unreadable letters
        scale = min(1000 / longest, 3.0)
    else:
        scale = 1.0
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        return image  # contrast is handled inside ocr.enhance()


def describe_image(image: Image.Image) -> str | None:
    """One-sentence caption from BLIP. Returns None if the vision model is unavailable."""
    global _vision
    try:
        with _vision_lock:  # one caption at a time
            if _vision is None:  # first use: loads the model (downloads ~1 GB once)
                from transformers import BlipForConditionalGeneration, BlipProcessor
                _vision = (BlipProcessor.from_pretrained(config.VISION_MODEL),
                           BlipForConditionalGeneration.from_pretrained(config.VISION_MODEL))
            processor, model = _vision
            import torch
            small = image.copy()
            small.thumbnail((768, 768))  # BLIP shrinks to 384 anyway; this just saves time
            inputs = processor(images=small, return_tensors="pt")
            with torch.no_grad():  # no training, saves memory
                output_ids = model.generate(**inputs, max_new_tokens=40)
            return processor.decode(output_ids[0], skip_special_tokens=True).strip() or None
    except Exception:  # never let the vision model break the upload
        logger.exception("Image captioning failed")
        return None


def classify_image(text: str) -> str:
    """Guess: prescription, lab_report, medical_report (other text) or medical_image (no text)."""
    lower = (text or "").lower()
    if len(lower.split()) < MIN_TEXT_WORDS:
        return "medical_image"
    rx = sum(1 for pattern in PRESCRIPTION_HINTS if re.search(pattern, lower))
    lab = sum(1 for pattern in LAB_HINTS if re.search(pattern, lower))
    if rx >= 2 and rx > lab:
        return "prescription"
    if lab >= 2:
        return "lab_report"
    if rx >= 2:
        return "prescription"
    return "medical_report"


def process_image(path) -> dict:
    """Image file -> OCR text, document type and (optionally) a vision caption."""
    path = Path(path)
    image = load_image(path)

    started = time.time()
    result = ocr.extract_text(prepare_for_ocr(image))  # all text lines
    ocr_seconds = round(time.time() - started, 2)

    image_type = classify_image(result["text"])
    caption, vision_seconds = None, 0.0
    mode = config.VISION_MODE
    if mode == "always" or (mode == "auto" and image_type == "medical_image"):  # "off" skips vision
        started = time.time()
        caption = describe_image(image)
        vision_seconds = round(time.time() - started, 2)

    logger.info("image processed type=%s lines=%s conf=%s ocr=%ss vision=%ss",
                image_type, result["line_count"], result["confidence"], ocr_seconds, vision_seconds)
    return {"image_type": image_type, "text": result["text"], "lines": result["lines"],
            "line_count": result["line_count"], "ocr_confidence": result["confidence"],
            "vision_caption": caption, "ocr_seconds": ocr_seconds, "vision_seconds": vision_seconds}