"""
Report analyzer: turns an uploaded image into medicines, diagnoses, dates,
observations and the doctor's written instructions.

Safety design (milestone rule: the bot never gives its own advice):
  * the LLM only COPIES what the OCR text says,
  * every extracted item is then CHECKED against the OCR text; anything not
    found in the text is dropped and reported as a warning,
  * the final summary is built by plain code, not by the LLM,
  * phone numbers / e-mails are masked before the text reaches the LLM.
"""

import difflib
import json
import logging
import re
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app import config, llm, prompts
from app.guardrails.file_guard import check_document_text
from app.guardrails.pii import mask_pii
from app.multimodal import image_processor

logger = logging.getLogger("mediassist.multimodal")

TYPE_NAMES = {"prescription": "Prescription", "lab_report": "Lab report",
              "medical_report": "Medical report", "medical_image": "Medical image (no readable text)"}
LOW_CONFIDENCE = 0.5  # below this average OCR confidence we warn the user

# Standard meanings of common prescription abbreviations (a dictionary lookup, not medical advice).
FREQUENCY_MEANINGS = {
    "od": "once a day", "bd": "twice a day", "bid": "twice a day", "tds": "three times a day",
    "tid": "three times a day", "qid": "four times a day", "sos": "only when needed", "hs": "at bedtime",
    "1-0-0": "morning only", "0-1-0": "afternoon only", "0-0-1": "night only", "1-0-1": "morning and night",
    "1-1-0": "morning and afternoon", "0-1-1": "afternoon and night", "1-1-1": "morning, afternoon and night",
}


# ---------- what the LLM must return ----------

class MedicineItem(BaseModel):
    name: str = Field(description="Medicine name as written")
    dosage: str = Field(default="", description="Amount per dose as written, e.g. 500 mg, or empty")
    frequency: str = Field(default="", description="How often as written, e.g. BD or 1-0-1, or empty")


class ImageExtraction(BaseModel):
    medicines: list[MedicineItem] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    doctor_instructions: list[str] = Field(default_factory=list)

    @field_validator("medicines", mode="before")
    @classmethod
    def _plain_names_ok(cls, value):  # some models return ["Paracetamol"] instead of objects
        return [{"name": item} if isinstance(item, str) else item for item in (value or [])]


# ---------- checking the LLM against the OCR text ----------

def supported(value: str, text: str) -> bool:
    """True if `value` really comes from `text` (small OCR spelling mistakes are tolerated)."""
    value = (value or "").strip()
    if not value:
        return True  # nothing to check
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())  # 11,500 -> 11500
    value_clean = re.sub(r"(?<=\d),(?=\d)", "", value.lower())
    flat = re.sub(r"\s+", "", value_clean)
    if len(flat) >= 4 and flat in re.sub(r"\s+", "", text):  # exact copy, spaces ignored
        return True
    if len(flat) < 4:  # short values (BD, OD, 1-0-1) must stand alone in the text
        pattern = r"(?<![a-z0-9])" + r"\s*".join(map(re.escape, flat)) + r"(?![a-z0-9])"
        return re.search(pattern, text) is not None
    numbers = re.findall(r"\d+(?:\.\d+)?", value_clean)
    text_numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
    if any(number not in text_numbers for number in numbers):  # every number must be in the text
        return False
    words = re.findall(r"[a-z]{4,}", value_clean)
    if not words:
        return bool(numbers)
    text_words = set(re.findall(r"[a-z]{3,}", text))
    found = sum(1 for w in words if w in text_words or difflib.get_close_matches(w, text_words, n=1, cutoff=0.85))
    return found / len(words) >= 0.6  # most words must be in the text


def explain_frequency(frequency: str) -> str:
    """'BD' -> 'twice a day'. Returns '' if the abbreviation is not in the dictionary."""
    flat = re.sub(r"\s+", "", (frequency or "").lower())
    if flat in FREQUENCY_MEANINGS:
        return FREQUENCY_MEANINGS[flat]
    for token in re.findall(r"[a-z]+", frequency.lower()):  # e.g. "1 tablet TDS"
        if token in FREQUENCY_MEANINGS:
            return FREQUENCY_MEANINGS[token]
    match = re.search(r"\d-\d-\d", flat)
    return FREQUENCY_MEANINGS.get(match.group(), "") if match else ""


def verify_extraction(raw: ImageExtraction, text: str, warnings: list[str]) -> dict:
    """Keep only what is supported by `text`. Adds a warning for everything dropped."""
    medicines = []
    for item in raw.medicines:
        name = item.name.strip()
        if not name or not supported(name, text):
            warnings.append(f"Removed medicine '{name}': it is not in the image text.")
            continue
        dosage, frequency = item.dosage.strip(), item.frequency.strip()
        if not supported(dosage, text):
            warnings.append(f"Dose for {name} could not be confirmed in the image, so it is left blank.")
            dosage = ""
        if not supported(frequency, text):
            warnings.append(f"Frequency for {name} could not be confirmed in the image, so it is left blank.")
            frequency = ""
        medicines.append({"name": name, "dosage": dosage, "frequency": frequency,
                          "frequency_plain": explain_frequency(frequency)})

    cleaned = {"medicines": medicines}
    for key in ("diagnoses", "dates", "observations", "doctor_instructions"):
        kept = []
        for line in getattr(raw, key):
            line = line.strip()
            if not line:
                continue
            if supported(line, text):
                kept.append(line)
            else:
                warnings.append(f"Removed from {key.replace('_', ' ')}: '{line[:60]}' is not in the image text.")
        cleaned[key] = kept
    return cleaned


# ---------- readable summary (built by code, never by the LLM) ----------

def build_summary(result: dict) -> str:
    lines = [f"Document type: {TYPE_NAMES.get(result['image_type'], result['image_type'])}"]
    if result["medicines"]:
        lines.append("Medicines (as written):")
        for number, med in enumerate(result["medicines"], 1):
            plain = f" ({med['frequency_plain']})" if med["frequency_plain"] else ""
            lines.append(f"  {number}. {med['name']} - {med['dosage'] or 'dose not written'}"
                         f" - {med['frequency'] or 'frequency not written'}{plain}")
    if result["diagnoses"]:
        lines.append("Diagnoses: " + "; ".join(result["diagnoses"]))
    if result["dates"]:
        lines.append("Dates: " + "; ".join(result["dates"]))
    if result["observations"]:
        lines.append("Observations:")
        lines += [f"  - {line}" for line in result["observations"]]
    if result["vision_caption"]:
        lines.append(f"Image description (automatic, may be inaccurate): {result['vision_caption']}")
    if result["doctor_instructions"]:
        lines.append("Instructions written by the doctor (not our own advice):")
        lines += [f"  - {line}" for line in result["doctor_instructions"]]
    if len(lines) == 1:
        lines.append("No medicines, diagnoses or observations could be read from this image.")
    lines.append("This was read automatically from the image. Please check it against the original.")
    return "\n".join(lines)


# ---------- main function ----------

def analyze_image(path) -> dict:
    """Run OCR (+ vision), extract the facts, verify them, save the result and return it."""
    started = time.time()
    path = Path(path)
    info = image_processor.process_image(path)

    text, masked = mask_pii(info["text"])  # PII never reaches the LLM, the logs or the saved file
    check = check_document_text(text)  # prompt injection hidden inside the image text
    if not check.ok:
        raise ValueError(check.message)

    warnings: list[str] = []
    if masked:
        warnings.append("Personal details (" + ", ".join(masked) + ") were hidden.")
    if info["image_type"] != "medical_image" and info["ocr_confidence"] < LOW_CONFIDENCE:
        warnings.append("The text was hard to read. Please check every value against the original.")

    found = {"medicines": [], "diagnoses": [], "dates": [], "observations": [], "doctor_instructions": []}
    llm_seconds = 0.0
    if info["image_type"] == "medical_image":  # nothing to extract from a picture without text
        warnings.append("No readable text found. Only a general description is possible; "
                        "images cannot be used to diagnose.")
    else:
        llm_started = time.time()
        try:
            raw = llm.invoke_structured(prompts.IMAGE_EXTRACTION_PROMPT, ImageExtraction,
                                        {"image_type": info["image_type"], "ocr_text": text[:6000]})
            found = verify_extraction(raw, text, warnings)
        except Exception:  # never show raw errors (they can contain keys or paths)
            logger.exception("Image extraction failed")
            warnings.append("Automatic extraction is not available right now. The text read from the "
                            "image is shown instead.")
        llm_seconds = round(time.time() - llm_started, 2)

    result = {
        "id": uuid4().hex, "file": path.name, "image_type": info["image_type"],
        "ocr_text": text, "ocr_confidence": info["ocr_confidence"], "line_count": info["line_count"],
        "vision_caption": info["vision_caption"], **found, "warnings": warnings,
        "has_evidence": bool(text.strip() or info["vision_caption"]),  # used by the answer gate (step 11)
        "timings": {"ocr": info["ocr_seconds"], "vision": info["vision_seconds"], "llm": llm_seconds,
                    "total": round(time.time() - started, 2)},
    }
    result["summary"] = build_summary(result)
    (config.EXTRACTIONS_DIR / f"{result['id']}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")  # saved so the chat can use it later
    return result


def get_extraction(extraction_id: str) -> dict | None:
    """Load a saved result by id (used by the chat in a later step). None if not found."""
    if not re.fullmatch(r"[0-9a-f]{32}", extraction_id or ""):  # blocks path tricks like ../
        return None
    path = config.EXTRACTIONS_DIR / f"{extraction_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None