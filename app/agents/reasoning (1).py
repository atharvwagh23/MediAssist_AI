"""
Reasoning Agent: combines documents + database records + image text into ONE grounded answer.

Steps:
  1. Turn every source into a labelled evidence block (readable text, not raw JSON,
     so "9.8 g/dL" stays together and the output guard can find doses in the evidence).
  2. If there is no evidence, or the role may not see the data, answer with a fixed message (no LLM).
  3. Otherwise ask the LLM to answer only from the evidence, with source labels.
"""

import json
import logging
import time

from langchain_core.prompts import ChatPromptTemplate

from app import prompts
from app.agents.state import AgentState, add_trace
from app.guardrails.output_guard import NO_EVIDENCE_MESSAGE
from app.llm import get_llm

logger = logging.getLogger("mediassist.agents")

MAX_BLOCK_CHARS = 3500  # per source, keeps the prompt (and the wait) small
NEED_ID_MESSAGE = "Please include the patient ID (for example P00123) so I can look up the records."


def _line(record: dict) -> str:
    """One database row as 'key: value; key: value'. Empty values are skipped."""
    return "; ".join(f"{k}: {v}" for k, v in record.items() if str(v).strip() not in ("", "not recorded"))


def _lab_line(r: dict) -> str:
    """Value and unit stay together ('9.8 g/dL') because the dose guard looks for exactly that."""
    unit = "" if r.get("result_unit") in (None, "not recorded") else r["result_unit"]
    reference = (f"reference {r['reference_min']}-{r['reference_max']} {unit}".strip()
                 if "not recorded" not in (r.get("reference_min"), r.get("reference_max")) else "")
    parts = [f"{r.get('test_date', '')} {r.get('test_name', '')} {r.get('result_value', '')} {unit}".strip(),
             reference, f"status: {r.get('result_status', '')}", f"lab: {r.get('lab_name', '')}",
             f"notes: {r.get('report_notes', '')}"]
    return "; ".join(p for p in parts if p and not p.endswith(": not recorded") and not p.endswith(": "))


def format_db(entry: dict) -> str:
    data = entry["data"]
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            for row in value:
                lines.append(_lab_line(row) if key == "lab_results" else f"{key}: " + _line(row))
        elif isinstance(value, dict):
            lines.append(f"{key}: " + _line(value))
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_evidence(state: AgentState) -> tuple[str, list]:
    """Return (evidence text, sources list). sources feed the answer's citations."""
    blocks, sources = [], []
    for chunk in state.get("doc_results", []):
        blocks.append(f"[Document: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}")
        sources.append({"type": "document", "source": chunk["source"], "page": chunk["page"],
                        "score": chunk["final_score"]})
    for entry in state.get("db_results", []):
        if entry["status"] != "ok":
            continue
        text = format_db(entry)
        if len(text) > MAX_BLOCK_CHARS:
            text = text[:MAX_BLOCK_CHARS] + "\n(more rows not shown)"
        blocks.append(f"[Database: {entry['tool']}]\n{text}")
        sources.append({"type": "database", "source": entry["tool"], "page": None,
                        "detail": entry["arguments"].get("patient_id", "")})
    image = state.get("image") or {}
    if image:
        text = image.get("summary") or ""
        if image.get("vision_caption"):
            text += f"\nGeneral picture description (not a diagnosis): {image['vision_caption']}"
        if image.get("ocr_text"):
            text += f"\nText read from the image:\n{image['ocr_text'][:2500]}"
        label = image.get("original_name") or image.get("file", "image")
        blocks.append(f"[Uploaded image: {label}]\n{text}")
        sources.append({"type": "image", "source": label, "page": None})
    return "\n\n".join(blocks), sources


def access_notes(state: AgentState) -> str:
    notes = [f"- Not allowed for this user's role: {e['tool']}" for e in state.get("db_results", [])
             if e["status"] == "denied"]
    notes += [f"- {e['tool']}: {e['message']}" for e in state.get("db_results", []) if e["status"] == "error"]
    notes += [f"- {t} needs a patient ID, but none was given." for t in state["plan"]["missing"]]
    return "\n".join(notes) or "(none)"


def _ask(question: str, evidence: str, notes: str) -> str:
    chain = ChatPromptTemplate.from_template(prompts.REASONING_PROMPT) | get_llm()
    return chain.invoke({"question": question, "evidence": evidence, "notes": notes}).content.strip()


def _fixed(state: AgentState, started: float, message: str, note: str) -> dict:
    return {"answer": message, "sources": [], "evidence": "", "is_system_message": True,
            "trace": add_trace(state, "reasoning", started, note)}


def reasoning_agent(state: AgentState) -> dict:
    started = time.time()
    evidence, sources = build_evidence(state)
    notes = access_notes(state)
    results = state.get("db_results", [])

    if not evidence:  # nothing to build an answer from: do not guess, do not call the LLM
        if results and all(e["status"] == "denied" for e in results):
            roles = state.get("role") or "unknown"
            return _fixed(state, started, f"Your role ({roles}) is not allowed to see this information.",
                          "role not allowed")
        if state["plan"]["missing"] and not state["plan"]["use_documents"]:
            return _fixed(state, started, NEED_ID_MESSAGE, "patient ID missing")
        errors = [e["message"] for e in results if e["status"] == "error"]
        return _fixed(state, started, errors[0] if errors else NO_EVIDENCE_MESSAGE, "no evidence")

    try:
        answer = _ask(state["question"], evidence, notes)
    except Exception:  # never show raw errors (they can contain keys or paths)
        logger.exception("Reasoning LLM failed")
        return _fixed(state, started, "The language model is not available right now. Please try again.",
                      "LLM failed")
    note = f"{len(sources)} sources, {len(evidence)} evidence chars, {len(answer)} answer chars"
    return {"answer": answer, "sources": sources, "evidence": evidence, "is_system_message": False,
            "trace": add_trace(state, "reasoning", started, note)}
