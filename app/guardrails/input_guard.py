"""
Input guardrail: runs on every user message BEFORE any agent.

Order of checks:
  1. empty or too long
  2. prompt injection / unsafe database requests  -> block
  3. PII masking (phone, e-mail, Aadhaar)         -> text is masked, flow continues
  4. vague message                                -> ask ONE clarifying question
  5. medical-scope check (keywords first, then a small LLM call) -> block if not medical

Result: GuardResult with action "allow", "block" or "clarify".
"""

import logging
import re
from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, Field

from app.guardrails.pii import mask_pii

logger = logging.getLogger("mediassist.guard")

MAX_QUESTION_CHARS = 2000

NON_MEDICAL_MESSAGE = (
    "I can only help with hospital and medical-related questions, such as admission, discharge, "
    "insurance, appointments, lab reports, prescriptions and hospital policies. Your request does "
    "not seem to be related to the medical field. Please ask a healthcare-related question."
)
INJECTION_MESSAGE = (
    "I can't follow that request because it tries to change my safety rules. "
    "Please ask a normal hospital or medical question."
)
UNSAFE_DB_MESSAGE = (
    "I can only read hospital data. I can't change, delete or run database commands. "
    "Tell me what information you want to look up instead."
)
CLARIFY_MESSAGE = (
    "Could you tell me a little more? For example, which patient ID, which document, "
    "or which process (admission, discharge, insurance) do you mean?"
)
VERIFY_FAILED_MESSAGE = (
    "I could not confirm that your question is about a hospital or medical topic. "
    "Please rephrase it and mention what you need (for example a patient ID, a report or a hospital process)."
)

INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(ignore|disregard|forget|override)\s+(?:(?:all|any|your|the|these|my)\s+)?(?:(?:previous|prior|above|earlier|system|safety|original)\s+)+(?:instructions?|prompts?|rules?|guidelines?|guardrails?|restrictions?)\b",
    r"\b(ignore|disregard|forget)\s+(?:(?:all|any|your)\s+)+(?:instructions?|rules?|guidelines?|guardrails?|restrictions?)\b",
    r"\b(bypass|disable|override|turn off)\s+(?:the\s+|your\s+)?(?:guardrails?|safety (?:rules|filters?|checks?)|restrictions?|filters?)\b",
    r"\b(reveal|show|print|repeat|display|leak)\b.{0,30}\b(system prompt|hidden prompt|your instructions|your prompt)\b",
    r"\bsystem prompt\b",
    r"\b(developer mode|jailbreak|jailbroken|do anything now)\b",
    r"\byou are now\b.{0,30}\b(dan|unrestricted|unfiltered|jailbroken|developer mode|no rules)\b",
    r"\bpretend\b.{0,30}\b(no|without)\b.{0,15}\b(rules|restrictions|limits|guardrails)\b",
    r"<\|?\s*(im_start|im_end|system)\s*\|?>|</?system>|\[/?INST\]",
    r"^\s*#{2,}\s*system\b",
)]
UNSAFE_DB_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(drop|truncate|alter)\s+(table|database|schema)\b",
    r"\bdelete\s+from\b",
    r"\binsert\s+into\b",
    r"\bupdate\s+\w+\s+set\b",
    r";\s*--",
)]

MEDICAL_WORDS = re.compile(
    r"\b(patients?|doctors?|nurses?|hospitals?|clinics?|medic\w*|medicines?|drugs?|tablets?|capsules?|"
    r"doses?|dosage|prescri\w*|diagnos\w*|symptoms?|diseases?|illness\w*|treatments?|therap\w*|surger\w*|"
    r"admission|admit\w*|discharge\w*|appointments?|labs?|laborator\w*|blood|reports?|scans?|x-?rays?|"
    r"mri|insurance|insurer|claims?|cashless|tpa|pre-?auth\w*|billing|bills?|refunds?|payments?|"
    r"sop|compliance|nabh|escalation|wards?|icu|emergency|opd|ipd|departments?|branch\w*|"
    r"specialists?|health\w*|medical|pharmac\w*|allerg\w*|diabet\w*|hba1c|fever|infection|vaccin\w*|"
    r"uhid|pregnan\w*|cardio\w*|oncolog\w*|policy|policies)\b",
    re.IGNORECASE,
)
PATIENT_ID_IN_TEXT = re.compile(r"\bP\d{3,10}\b", re.IGNORECASE)

# Tools an agent may call, and what their arguments may look like.
ALLOWED_TOOLS = {"search_patients", "get_patient_history", "get_appointments",
                 "get_lab_results", "get_billing_summary", "get_patient_summary", "get_prescriptions",
                 "get_admissions", "get_doctor_schedule", "search_doctors", "search_lab_tests",
                 "get_insurance_details", "get_department_statistics"}


@dataclass
class GuardResult:
    action: str                      # "allow", "block" or "clarify"
    reason: str = ""                 # short code: ok, empty, too_long, prompt_injection, ...
    message: str = ""                # text to show the user when not allowed
    masked_text: str = ""            # the question with PII masked (use THIS for the LLM)
    pii_found: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    def to_dict(self) -> dict:
        return asdict(self)


class ScopeDecision(BaseModel):
    is_medical: bool = Field(description="True if the message is about the medical or hospital field")
    reason: str = Field(default="", description="One short reason")


def find_injection(text: str) -> str | None:
    """Return a short label if the text looks like a prompt-injection attempt."""
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text or ""):
            return "prompt_injection"
    return None


def classify_scope(question: str, history: list[str] | None = None) -> bool:
    """Ask a small LLM call: is this medical/hospital related? Raises if the call fails."""
    from app import prompts
    from app.llm import invoke_structured

    recent = "\n".join(f"- {h[:300]}" for h in (history or [])[-2:]) or "(none)"
    decision = invoke_structured(prompts.SCOPE_CLASSIFIER_PROMPT, ScopeDecision,
                                 {"history": recent, "question": question})
    return bool(decision.is_medical)

def check_input(text: str, has_upload: bool = False, history: list[str] | None = None) -> GuardResult:
    """Run every input check. history = earlier user messages (for follow-up questions)."""
    text = (text or "").strip()
    if not text:
        return GuardResult("clarify", "empty", CLARIFY_MESSAGE, "")
    if len(text) > MAX_QUESTION_CHARS:
        return GuardResult("block", "too_long",
                           f"Please keep your question under {MAX_QUESTION_CHARS} characters.", "")

    if find_injection(text):
        return GuardResult("block", "prompt_injection", INJECTION_MESSAGE, "")
    if any(p.search(text) for p in UNSAFE_DB_PATTERNS):
        return GuardResult("block", "unsafe_database_request", UNSAFE_DB_MESSAGE, "")

    masked, pii = mask_pii(text)  # from here on, only the masked text is used

    has_medical_word = bool(MEDICAL_WORDS.search(masked)) or bool(PATIENT_ID_IN_TEXT.search(masked))
    if len(masked.split()) <= 2 and not has_medical_word and not has_upload:
        return GuardResult("clarify", "vague", CLARIFY_MESSAGE, masked, pii)

    if has_medical_word or has_upload:  # fast path: no LLM call needed
        return GuardResult("allow", "ok", "", masked, pii)

    try:  # ambiguous: let the small LLM decide
        medical = classify_scope(masked, history)
    except Exception:
        logger.exception("Scope classifier failed")
        return GuardResult("clarify", "scope_check_failed", VERIFY_FAILED_MESSAGE, masked, pii)
    if not medical:
        return GuardResult("block", "not_medical", NON_MEDICAL_MESSAGE, masked, pii)
    return GuardResult("allow", "ok", "", masked, pii)


def check_tool_call(name: str, arguments: dict) -> tuple[bool, str]:
    """Tool-misuse guard: only allowlisted tools, only plain short string/number/bool arguments."""
    if name not in ALLOWED_TOOLS:
        return False, f"Tool '{name}' is not allowed."
    for key, value in (arguments or {}).items():
        if not isinstance(value, (str, int, float, bool)):
            return False, f"Argument '{key}' has an unsupported type."
        if isinstance(value, str):
            if len(value) > 200:
                return False, f"Argument '{key}' is too long."
            if any(p.search(value) for p in UNSAFE_DB_PATTERNS):
                return False, f"Argument '{key}' contains database commands."
    return True, ""
