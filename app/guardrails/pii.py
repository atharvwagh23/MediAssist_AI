"""
PII masking with plain regular expressions (no extra library).

Always masked: phone numbers, e-mail addresses, Aadhaar-style 12-digit numbers.
Patient IDs (P00001) are NOT masked in chat text, because the database tools
need them to look a patient up. They are masked only when text goes to logs
(mask_for_log).
"""

import re

PATTERNS = [  # order matters: the longer 12-digit pattern must run before the phone one
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
]
PATIENT_ID = ("PATIENT_ID", re.compile(r"\bP\d{3,10}\b", re.IGNORECASE))


def mask_pii(text: str, include_ids: bool = False) -> tuple[str, list[str]]:
    """Return (masked text, list of PII types found). Example: '[PHONE]'."""
    found: list[str] = []
    patterns = PATTERNS + ([PATIENT_ID] if include_ids else [])
    for label, pattern in patterns:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label}]", text)
    return text, found


def mask_for_log(text: str) -> str:
    """Use this before writing any user text to a log or trace."""
    return mask_pii(text or "", include_ids=True)[0]
