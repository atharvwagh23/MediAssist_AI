"""
Evaluator Agent: the last check before the user sees an answer.

  1. Rule checks (output guardrail): PII masked, doses must be in the evidence, no own advice.
  2. LLM-as-a-Judge (JUDGE_MODEL): faithfulness, grounding, relevance, hallucination (1 to 5 / yes-no).
  3. Retrieval quality from the real scores (document scores, tool success, OCR confidence).
  4. Decision: block if hallucination, weak faithfulness/grounding, own treatment advice or off scope.

If the judge call fails, the rule checks still apply and the result says "judge unavailable".
"""

import logging
import time

from pydantic import BaseModel, Field

from app import config, prompts
from app.agents.state import AgentState, add_trace
from app.guardrails.output_guard import OFF_SCOPE_MESSAGE, UNSAFE_MESSAGE, check_output
from app.llm import invoke_structured

logger = logging.getLogger("mediassist.agents")

MIN_SCORE = 3  # faithfulness and grounding below this are blocked
LOW_CONFIDENCE = 0.7  # below this the answer is shown with a warning
LOW_CONFIDENCE_NOTE = "\n\n*Confidence is low. Please check this against the original records.*"


class JudgeScores(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="1 to 5: claims supported by the evidence")
    grounding: int = Field(ge=1, le=5, description="1 to 5: facts carry correct source labels")
    relevance: int = Field(ge=1, le=5, description="1 to 5: answers the question")
    hallucination_detected: bool = Field(description="A medicine, dose, value or name not in the evidence")
    adds_own_treatment: bool = Field(description="Recommends treatment not written in the evidence")
    in_medical_scope: bool = Field(default=True, description="About the medical or hospital field")
    reason: str = Field(default="", description="One short reason")


def retrieval_quality(state: AgentState) -> float | None:
    """0 to 1: how good was the evidence? Average over the sources that were used."""
    parts = []
    docs = state.get("doc_results", [])
    if docs:
        parts.append(sum(d["final_score"] for d in docs) / len(docs))
    calls = [e for e in state.get("db_results", []) if e["status"] != "denied"]
    if calls:
        parts.append(sum(1 for e in calls if e["status"] == "ok") / len(calls))
    image = state.get("image") or {}
    if image:
        parts.append(float(image.get("ocr_confidence") or 0))
    return round(min(sum(parts) / len(parts), 1.0), 3) if parts else None


def judge(question: str, evidence: str, answer: str) -> JudgeScores:
    return invoke_structured(prompts.JUDGE_PROMPT, JudgeScores,
                             {"question": question, "evidence": evidence[:7000], "answer": answer[:4000]},
                             model=config.JUDGE_MODEL)


def _result(state: AgentState, started: float, text: str, allowed: bool, evaluation: dict) -> dict:
    note = ("allowed" if allowed else "blocked") + f", confidence={evaluation.get('confidence')}"
    return {"final_answer": text, "allowed": allowed, "evaluation": evaluation,
            "trace": add_trace(state, "evaluator", started, note)}


def evaluator_agent(state: AgentState) -> dict:
    started = time.time()
    quality = retrieval_quality(state)
    base = {"retrieval_quality": quality, "guard_issues": [], "judge": "skipped", "confidence": None}

    if state.get("is_system_message"):  # fixed message (no evidence / not allowed): nothing to judge
        return _result(state, started, state["answer"], False, {**base, "decision": "no_answer"})

    guard = check_output(state["answer"], state["sources"], evidence=state["evidence"], use_llm=False)
    base["guard_issues"] = guard.issues
    if not guard.allowed:
        return _result(state, started, guard.answer, False, {**base, "decision": "blocked_by_rules"})

    try:
        scores = judge(state["question"], state["evidence"], guard.answer)
    except Exception:
        logger.exception("Judge failed; rule checks were still applied")
        base["judge"] = "unavailable"
        return _result(state, started, guard.answer, True, {**base, "decision": "allowed_rules_only"})

    judge_part = (scores.faithfulness + scores.grounding + scores.relevance) / 15
    confidence = round(0.8 * judge_part + 0.2 * (quality if quality is not None else judge_part), 2)
    evaluation = {**base, "judge": "ok", "faithfulness": scores.faithfulness, "grounding": scores.grounding,
                  "relevance": scores.relevance, "hallucination": scores.hallucination_detected,
                  "confidence": confidence, "judge_reason": scores.reason[:300]}

    if not scores.in_medical_scope:
        return _result(state, started, OFF_SCOPE_MESSAGE, False, {**evaluation, "decision": "blocked_off_scope"})
    if (scores.hallucination_detected or scores.adds_own_treatment
            or scores.faithfulness < MIN_SCORE or scores.grounding < MIN_SCORE):
        return _result(state, started, UNSAFE_MESSAGE, False, {**evaluation, "decision": "blocked_by_judge"})

    text = guard.answer + (LOW_CONFIDENCE_NOTE if confidence < LOW_CONFIDENCE else "")
    return _result(state, started, text, True, {**evaluation, "decision": "allowed"})
