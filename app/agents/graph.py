"""
LangGraph orchestrator: wires the six agents into one workflow.

    START -> planner -> (retriever | mcp | vision, in parallel) -> reasoning -> evaluator -> END

The three data agents do not depend on each other, so they run at the same time.
That saves several seconds compared with running them one after the other.

Every node is wrapped so that:
  * a crash inside one agent never crashes the chat (the user gets a safe message), and
  * each agent adds only its own line to the trace (parallel agents cannot overwrite each other).
"""

import logging
import time

from langgraph.graph import END, START, StateGraph

from app.agents import evaluator, mcp_agent, planner, reasoning, retriever_agent, vision_agent
from app.agents.state import AgentState

logger = logging.getLogger("mediassist.agents")

FAILED_REASONING = {"answer": "Something went wrong while preparing the answer. Please try again.",
                    "sources": [], "evidence": "", "is_system_message": True}
FAILED_EVALUATOR = {"final_answer": "Something went wrong while checking the answer. Please try again.",
                    "allowed": False, "evaluation": {"decision": "error", "confidence": None}}
FAILED = {  # what each agent returns if it crashes (empty results, so the next agent can still run)
    "retriever": {"doc_results": []}, "mcp": {"db_results": []}, "vision": {"image": {}},
    "reasoning": FAILED_REASONING, "evaluator": FAILED_EVALUATOR,
}


def _node(name: str, agent):
    def run(state: AgentState) -> dict:
        started = time.time()
        before = len(state.get("trace", []))
        try:
            out = dict(agent(state))
        except Exception:
            logger.exception("Agent %s crashed", name)
            fallback = {**FAILED.get(name, {})}
            fallback["trace"] = [{"agent": name, "seconds": round(time.time() - started, 2), "note": "FAILED"}]
            return fallback
        out["trace"] = out.get("trace", [])[before:]  # only this agent's own new line
        return out
    return run


def build_graph():
    """Create and compile the graph. Agent functions are looked up here (tests can swap them first)."""
    graph = StateGraph(AgentState)
    graph.add_node("planner", _node("planner", planner.planner_agent))
    graph.add_node("retriever", _node("retriever", retriever_agent.retriever_agent))
    graph.add_node("mcp", _node("mcp", mcp_agent.mcp_agent))
    graph.add_node("vision", _node("vision", vision_agent.vision_agent))
    graph.add_node("reasoning", _node("reasoning", reasoning.reasoning_agent))
    graph.add_node("evaluator", _node("evaluator", evaluator.evaluator_agent))

    graph.add_edge(START, "planner")
    for name in ("retriever", "mcp", "vision"):  # fan out
        graph.add_edge("planner", name)
    graph.add_edge(["retriever", "mcp", "vision"], "reasoning")  # fan in: waits for all three
    graph.add_edge("reasoning", "evaluator")
    graph.add_edge("evaluator", END)
    return graph.compile()


_graph = None  # built once, on the first question


def run_graph(question: str, extraction_id: str = "") -> dict:
    """Run all six agents and return the final state (final_answer, allowed, evaluation, trace...)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    started = time.time()
    state = _graph.invoke({"question": question, "extraction_id": extraction_id, "trace": []})
    state["total_seconds"] = round(time.time() - started, 2)
    return state
