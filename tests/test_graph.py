"""
Graph test: the six agents are replaced by tiny fakes, so we test only the WIRING
(order, parallel merge, crash safety). No database, no LLM, no internet.

Run from the project root:
    python -m tests.test_graph
"""

import logging

from app.agents import evaluator, graph, mcp_agent, planner, reasoning, retriever_agent, vision_agent

logging.disable(logging.CRITICAL)  # crashes below are on purpose
REAL_EVALUATOR = evaluator.evaluator_agent  # kept, so one test can use the real safety rules
results = []  # (test name, passed?)


def check(name: str, condition: bool) -> None:  # record and print one result
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}")


def line(state, agent, note="ok"):  # what a real agent returns for the trace
    return list(state.get("trace", [])) + [{"agent": agent, "seconds": 0.0, "note": note}]


received = {}


def fake_planner(state):
    received.update(role=state["role"], question=state["question"], extraction_id=state["extraction_id"])
    return {"plan": {"use_documents": True, "calls": [], "missing": []}, "trace": line(state, "planner")}


def fake_retriever(state):
    return {"doc_results": [{"source": "a.pdf", "page": 1, "text": "t", "final_score": 0.8}],
            "trace": line(state, "retriever")}


def fake_mcp(state):
    return {"db_results": [{"tool": "get_lab_results", "status": "ok"}], "trace": line(state, "mcp")}


def fake_vision(state):
    return {"image": {"file": "rx.png"}, "trace": line(state, "vision")}


def fake_reasoning(state):  # proves the three parallel agents' results were all merged
    merged = bool(state.get("doc_results")) and bool(state.get("db_results")) and bool(state.get("image"))
    return {"answer": "merged" if merged else "MISSING DATA", "sources": [{"source": "a.pdf"}], "evidence": "e",
            "is_system_message": False, "trace": line(state, "reasoning")}


def fake_evaluator(state):
    return {"final_answer": state["answer"], "allowed": True, "evaluation": {"decision": "allowed"},
            "trace": line(state, "evaluator")}


def install(**overrides):
    fakes = {"planner": fake_planner, "retriever_agent": fake_retriever, "mcp_agent": fake_mcp,
             "vision_agent": fake_vision, "reasoning_agent": fake_reasoning, "evaluator_agent": fake_evaluator}
    fakes.update(overrides)
    planner.planner_agent = fakes["planner"]
    retriever_agent.retriever_agent = fakes["retriever_agent"]
    mcp_agent.mcp_agent = fakes["mcp_agent"]
    vision_agent.vision_agent = fakes["vision_agent"]
    reasoning.reasoning_agent = fakes["reasoning_agent"]
    evaluator.evaluator_agent = fakes["evaluator_agent"]
    graph._graph = None  # rebuild with the fakes


install()
state = graph.run_graph("Show labs for P00001", "doctor", "a" * 32)
agents = [t["agent"] for t in state["trace"]]
check("graph: all six agents ran, once each", sorted(agents) == sorted(
    ["planner", "retriever", "mcp", "vision", "reasoning", "evaluator"]) and len(agents) == 6)
check("graph: planner runs first, evaluator last", agents[0] == "planner" and agents[-1] == "evaluator")
check("graph: reasoning runs after the three data agents", agents.index("reasoning") > max(
    agents.index("retriever"), agents.index("mcp"), agents.index("vision")))
check("graph: results of the three parallel agents are all merged", state["final_answer"] == "merged")
check("graph: question, role and image id reach the planner",
      received == {"role": "doctor", "question": "Show labs for P00001", "extraction_id": "a" * 32})
check("graph: final state has allowed, evaluation and total time",
      state["allowed"] is True and state["evaluation"]["decision"] == "allowed" and "total_seconds" in state)


def crash(state):
    raise RuntimeError("secret key abc123")


install(retriever_agent=crash)
state = graph.run_graph("q", "doctor")
check("graph: a crashing data agent does not stop the others", state["allowed"] is True)
check("graph: the crash is shown in the trace as FAILED",
      any(t["agent"] == "retriever" and t["note"] == "FAILED" for t in state["trace"]))
install(reasoning_agent=crash, evaluator_agent=REAL_EVALUATOR)  # the real evaluator never lets a fixed message through
state = graph.run_graph("q", "doctor")
check("graph: a crashing reasoning agent gives a safe message, no stack trace",
      "abc123" not in str(state) and "Something went wrong" in state["final_answer"])
check("graph: ...and it is not shown as a successful answer", state["allowed"] is False)
install(evaluator_agent=crash)
state = graph.run_graph("q", "doctor")
check("graph: a crashing evaluator blocks the answer (never lets it through unchecked)",
      state["allowed"] is False and "Something went wrong" in state["final_answer"])

passed = sum(1 for _, ok in results if ok)
print(f"\n{passed}/{len(results)} tests passed")
raise SystemExit(0 if passed == len(results) else 1)
