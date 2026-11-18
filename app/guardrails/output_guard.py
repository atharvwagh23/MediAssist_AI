"""
Output guardrail: runs on the final answer BEFORE the user sees it.

Checks:
  1. empty answer
  2. no evidence/source behind the answer          -> replaced by "no information"
  3. PII in the answer (phone, e-mail, Aadhaar)    -> masked
  4. doses (mg, ml, ...) that are NOT in the evidence -> blocked
  5. first-person treatment advice                 -> blocked
  6. LLM review: still medical? adds its own treatment? (skipped safely if the call fails)
  7. adds the "consult your doctor" note

The rule-based checks (2 to 5) always run. Check 6 is an extra layer.
"""

import logging
import re
from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, Field

from app.guardrails.pii import mask_pii

logger = logging.getLogger("mediassist.guard")

DISCLAIMER = "\n\n*This information is for reference only and does not replace advice from your doctor.*"
NO_EVIDENCE_MESSAGE = (
    "I couldn't find information about this in the hospital knowledge base, the patient records "
    "or your uploaded files. Please rephrase your question or contact the hospital help desk."
)
UNSAFE_MESSAGE = (
    "I can't provide that answer because it contains medical advice that is not supported by the "
    "hospital documents or the patient's records. Please ask your doctor."
)
OFF_SCOPE_MESSAGE = (
    "I can only help with hospital and medical-related questions. Please ask a healthcare-related question."
)

DOSE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|iu|units?)\b", re.IGNORECASE)
OWN_ADVICE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\bI\s+(?:would\s+)?(?:recommend|suggest|advise|prescribe)\b",
    r"\byou\s+(?:should|must|need to|ought to)\s+(?:take|start|stop|increase|decrease|double|skip|avoid|use)\b",
    r"\b(?:try|consider)\s+taking\b",
)]


class OutputReview(BaseModel):
    in_medical_scope: bool = Field(description="True if the answer is about the medical or hospital field")
    adds_own_treatment: bool = Field(description="True if it recommends a medicine, dose or treatment not in the evidence")
    reason: str = Field(default="", description="One short reason")


@dataclass
class OutputResult:
    action: str                      # "allow" or "block"
    answer: str                      # text to show the user (final)
    issues: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def unsupported_doses(answer: str, evidence: str) -> list[str]:
    """Doses in the answer that do not appear in the evidence (spaces and case ignored)."""
    evidence_flat = _normalize(evidence)
    return sorted({d for d in DOSE_PATTERN.findall(answer) if _normalize(d) not in evidence_flat})


def review_with_llm(answer: str, evidence: str) -> OutputReview:
    """Small LLM call with a fixed output format. Raises if the call fails."""
    from app import prompts
    from app.llm import invoke_structured

    return invoke_structured(prompts.OUTPUT_REVIEW_PROMPT, OutputReview,
                             {"answer": answer[:4000], "evidence": evidence[:6000] or "(none)"})


def _block(message: str, issues: list[str]) -> OutputResult:
    return OutputResult("block", message, issues)


def check_output(answer: str, sources: list | None, evidence: str = "", use_llm: bool = True) -> OutputResult:
    """sources = citations behind the answer, evidence = the text the answer was built from."""
    answer = (answer or "").strip()
    issues: list[str] = []
    if not answer:
        return _block(NO_EVIDENCE_MESSAGE, ["empty_answer"])
    if not sources:
        return _block(NO_EVIDENCE_MESSAGE, ["no_source"])

    answer, pii = mask_pii(answer)
    if pii:
        issues.append("pii_masked:" + ",".join(pii))

    bad_doses = unsupported_doses(answer, evidence)
    if bad_doses:
        return _block(UNSAFE_MESSAGE, issues + ["dose_not_in_sources:" + ",".join(bad_doses)])
    if any(p.search(answer) for p in OWN_ADVICE_PATTERNS):
        return _block(UNSAFE_MESSAGE, issues + ["own_treatment_advice"])

    if use_llm:
        try:
            review = review_with_llm(answer, evidence)
            if not review.in_medical_scope:
                return _block(OFF_SCOPE_MESSAGE, issues + ["off_scope"])
            if review.adds_own_treatment:
                return _block(UNSAFE_MESSAGE, issues + ["llm_own_treatment"])
        except Exception:
            logger.exception("Output review failed; rule-based checks were still applied")
            issues.append("llm_review_skipped")

    if "does not replace advice from your doctor" not in answer:
        answer += DISCLAIMER
    return OutputResult("allow", answer, issues)
