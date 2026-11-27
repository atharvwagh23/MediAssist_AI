"""
The shared state that flows through all six agents.

Every agent is a plain function: agent(state) -> dict of the keys it changes.
In step 12 LangGraph calls these same functions as nodes, so nothing here changes.
"""

import operator
import time
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # ----- input (set by the API before the graph starts) -----
    question: str          # already checked and PII-masked by the input guardrail
    extraction_id: str     # id of an uploaded image result (optional)
    # ----- Planner -----
    plan: dict
    # ----- Retriever / MCP / Vision -----
    doc_results: list      # chunks from the vector search
    db_results: list       # one entry per MCP tool call
    image: dict            # saved result of an uploaded image
    # ----- Reasoning -----
    answer: str
    sources: list
    evidence: str
    is_system_message: bool  # True when the answer is a fixed message (no LLM, nothing to judge)
    # ----- Evaluator -----
    evaluation: dict
    final_answer: str
    allowed: bool
    # ----- tracing for the dashboard -----
    trace: Annotated[list, operator.add]  # agents running in parallel each ADD their own line


def add_trace(state: AgentState, agent: str, started: float, note: str) -> list:
    """Return the trace list with one more line: which agent ran, how long, what it did."""
    return list(state.get("trace", [])) + [
        {"agent": agent, "seconds": round(time.time() - started, 2), "note": note}
    ]
