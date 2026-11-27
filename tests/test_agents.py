"""
Agent tests. No database, no LLM and no internet are used: every outside call is replaced
by a small fake, so the tests are fast and the same every time.

Run from the project root:
    python -m tests.test_agents
"""

import logging

from app.agents import evaluator, mcp_agent, planner, reasoning, retriever_agent, vision_agent

logging.disable(logging.CRITICAL)  # the tests make failures on purpose; keep the output clean
results = []  # (test name, passed?)


def check(name: str, condition: bool) -> None:  # record and print one result
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}")


def boom(*args, **kwargs):  # a fake that always fails
    raise RuntimeError("simulated failure")


PLAN_FORM = planner.Plan  # shortcut


def plan_for(question, llm_plan=None, image=None):
    """Run the Planner with a fake LLM (or a failing one when llm_plan is None)."""
    planner.invoke_structured = (lambda *a, **k: llm_plan) if llm_plan else boom
    state = {"question": question, "image": image or {}, "extraction_id": "x" * 32 if image else ""}
    return planner.planner_agent(state)["plan"]


# ---- 1. Planner ----
p = plan_for("Show lab results for P00123", PLAN_FORM(use_database=True, tools=["get_lab_results"], patient_id="p00123"))
check("planner: LLM plan becomes a tool call with an upper-case ID",
      p["calls"] == [{"name": "get_lab_results", "arguments": {"patient_id": "P00123"}}] and p["source"] == "llm")
p = plan_for("Lab results for P00123", PLAN_FORM(use_database=True, tools=["drop_table", "get_lab_results", "get_lab_results"], patient_id="P00123"))
check("planner: unknown tools are dropped, duplicates removed", [c["name"] for c in p["calls"]] == ["get_lab_results"])
p = plan_for("History of P00123", PLAN_FORM(use_database=True, tools=["get_patient_history"], patient_id="P99999"))
check("planner: the patient ID in the question wins over the LLM's", p["calls"][0]["arguments"]["patient_id"] == "P00123")
p = plan_for("Show history", PLAN_FORM(use_database=True, tools=["get_patient_history"], patient_id="1; DROP TABLE patients"))
check("planner: a bad patient ID from the LLM is ignored and reported missing",
      p["calls"] == [] and p["missing"] == ["get_patient_history"])
p = plan_for("What is the refund process?", PLAN_FORM(use_documents=True))
check("planner: a document question plans no database call", p["use_documents"] and not p["calls"])
p = plan_for("hello there", PLAN_FORM())
check("planner: an empty LLM plan falls back to documents", p["use_documents"])
p = plan_for("Abnormal labs and upcoming appointments for P00123",
             PLAN_FORM(use_database=True, tools=["get_lab_results", "get_appointments"], abnormal_only=True, upcoming_only=True))
check("planner: filters reach the tool arguments",
      p["calls"][0]["arguments"].get("abnormal_only") is True and p["calls"][1]["arguments"].get("upcoming_only") is True)
p = plan_for("Find patient named Rahul Sharma in Pune", PLAN_FORM(use_database=True, tools=["search_patients"], search_name="Rahul Sharma", search_city="Pune"))
check("planner: search_patients gets name and city",
      p["calls"][0]["arguments"] == {"name": "Rahul Sharma", "city": "Pune"})
image = {"ocr_text": "Patient ID: P00777 Date: 2026-10-14", "file": "lab.png"}
p = plan_for("Get history for this report", PLAN_FORM(use_database=True, use_image=True, tools=["get_patient_history"]), image=image)
check("planner: patient ID is taken from the image when the question has none",
      p["calls"][0]["arguments"]["patient_id"] == "P00777" and p["use_image"])

# fallback rules (LLM down)
p = plan_for("Show appointments for P00005")
check("planner rules: LLM failure still plans the right tool", p["source"] == "rules" and p["calls"][0]["name"] == "get_appointments")
p = plan_for("What is the admission process?")
check("planner rules: a process question uses documents", p["use_documents"] and not p["calls"])
p = plan_for("Show billing and lab results for P00009")
check("planner rules: two topics give two tools", [c["name"] for c in p["calls"]] == ["get_lab_results", "get_billing_summary"])
p = plan_for("Show lab results")
check("planner rules: no patient ID gives 'missing', not a call", p["calls"] == [] and p["missing"] == ["get_lab_results"])

# new tools
p = plan_for("Which department had the most admissions in 2025?",
             PLAN_FORM(use_database=True, tools=["get_department_statistics"], metric="admissions_by_department", year="2025"))
check("planner: statistics tool needs no patient ID",
      p["calls"] == [{"name": "get_department_statistics", "arguments": {"metric": "admissions_by_department", "year": "2025"}}])
p = plan_for("Prescriptions for P00001", PLAN_FORM(use_database=True, tools=["get_prescriptions"], patient_id="P00001", medicine="met"))
check("planner: medicine filter reaches get_prescriptions", p["calls"][0]["arguments"] == {"patient_id": "P00001", "medicine": "met"})
p = plan_for("Find cardiologists in Pune", PLAN_FORM(use_database=True, tools=["search_doctors"], specialization="Cardiology", search_city="Pune"))
check("planner: search_doctors gets specialization and city", p["calls"][0]["arguments"] == {"specialization": "Cardiology", "city": "Pune"})
p = plan_for("Admissions of this patient", PLAN_FORM(use_database=True, tools=["get_admissions"]))
check("planner: patient tools without an ID are 'missing'", p["calls"] == [] and p["missing"] == ["get_admissions"])
p = plan_for("Statistics", PLAN_FORM(use_database=True, tools=["get_department_statistics"], metric="drop table", year="20x5"))
check("planner: bad metric and year are cleaned", p["calls"][0]["arguments"] == {"metric": "admissions_by_department"})
p = plan_for("Which department had the most admissions in 2025?")
check("planner rules: hospital-wide question uses statistics, not patient tools",
      [c["name"] for c in p["calls"]] == ["get_department_statistics"] and p["calls"][0]["arguments"].get("year") == "2025")
p = plan_for("Show prescriptions for P00003")
check("planner rules: prescriptions keyword", p["calls"][0]["name"] == "get_prescriptions")

# ---- 2. MCP agent ----
calls_made = []


def fake_tool(name, arguments):
    calls_made.append(name)
    if name == "get_billing_summary":
        return {"error": "No patient with ID P00001."}
    return {"patient_id": arguments.get("patient_id"), "count": 1}


mcp_agent.client.call_tool_sync = fake_tool
two = {"plan": {"calls": [{"name": "get_lab_results", "arguments": {"patient_id": "P00001"}},
                          {"name": "get_billing_summary", "arguments": {"patient_id": "P00001"}}], "missing": []}}
out = mcp_agent.mcp_agent({**two})["db_results"]
check("mcp: both tools run, order is kept", [r["tool"] for r in out] == ["get_lab_results", "get_billing_summary"])
check("mcp: a tool message becomes status 'error' with the message", out[1]["status"] == "error" and "No patient" in out[1]["message"])
bad = {"plan": {"calls": [{"name": "search_patients", "arguments": {"name": "x'; DROP TABLE patients;--"}}],
                "missing": []}}
mcp_agent.client.call_tool_sync = fake_tool
out = mcp_agent.mcp_agent({**bad})["db_results"]
check("mcp: SQL words in an argument are blocked by the tool guard", out[0]["status"] == "blocked")
mcp_agent.client.call_tool_sync = boom
out = mcp_agent.mcp_agent({**two})["db_results"]
check("mcp: a crash becomes a safe message (no stack trace)",
      out[0]["status"] == "error" and "simulated" not in out[0]["message"])
out = mcp_agent.mcp_agent({"plan": {"calls": [], "missing": []}})
check("mcp: no calls planned gives an empty result", out["db_results"] == [])

# ---- 3. Retriever and Vision agents ----
retriever_agent.retrieve = lambda q: {"found": True, "results": [
    {"source": "refund.pdf", "page": 2, "text": "Refunds take 7 days.", "final_score": 0.71}]}
out = retriever_agent.retriever_agent({"question": "refund?", "plan": {"use_documents": True}})
check("retriever: returns chunks when the plan asks for documents", out["doc_results"][0]["source"] == "refund.pdf")
out = retriever_agent.retriever_agent({"question": "x", "plan": {"use_documents": False}})
check("retriever: skipped when the plan does not need documents", out["doc_results"] == [])
retriever_agent.retrieve = lambda q: {"found": False, "results": []}
out = retriever_agent.retriever_agent({"question": "x", "plan": {"use_documents": True}})
check("retriever: nothing relevant gives an empty list", out["doc_results"] == [])

saved = {"file": "rx.png", "image_type": "prescription", "medicines": [{"name": "Paracetamol"}],
         "observations": [], "summary": "Document type: Prescription", "ocr_text": "Tab Paracetamol 500 mg BD"}
vision_agent.get_extraction = lambda i: saved if i == "a" * 32 else None
out = vision_agent.vision_agent({"plan": {"use_image": True}, "extraction_id": "a" * 32})
check("vision: loads the saved image result", out["image"]["file"] == "rx.png")
out = vision_agent.vision_agent({"plan": {"use_image": True}, "extraction_id": "b" * 32})
check("vision: unknown id gives an empty image, no crash", out["image"] == {})
out = vision_agent.vision_agent({"plan": {"use_image": False}})
check("vision: skipped when the plan has no image", "image" not in out)

# ---- 4. Reasoning agent ----
lab = {"tool": "get_lab_results", "arguments": {"patient_id": "P00001"}, "status": "ok", "data": {
    "patient_id": "P00001", "count": 1, "lab_results": [{
        "test_name": "Hemoglobin", "test_date": "2026-10-14", "result_value": 9.8, "result_unit": "g/dL",
        "reference_min": 12.0, "reference_max": 16.0, "result_status": "Low", "lab_name": "City Lab",
        "report_notes": "not recorded", "test_category": "CBC"}]}}
state = {"question": "Show labs for P00001", "doc_results": [], "db_results": [lab],
         "plan": {"missing": [], "use_documents": False}}
evidence, sources = reasoning.build_evidence(state)
check("reasoning: value and unit stay together ('9.8 g/dL')", "9.8 g/dL" in evidence and "12.0-16.0 g/dL" in evidence)
check("reasoning: database source is listed", sources[0]["type"] == "database" and sources[0]["detail"] == "P00001")
check("reasoning: 'not recorded' fields are left out", "not recorded" not in evidence)

reasoning._ask = lambda q, e, n: "Hemoglobin was 9.8 g/dL, flagged Low [Database: get_lab_results]."
out = reasoning.reasoning_agent(state)
check("reasoning: LLM answer is returned with evidence and sources",
      "9.8 g/dL" in out["answer"] and out["sources"] and not out["is_system_message"])
reasoning._ask = boom
out = reasoning.reasoning_agent(state)
check("reasoning: LLM failure gives a safe fixed message", out["is_system_message"] and "not available" in out["answer"])
reasoning._ask = lambda *a: "should not be called"
out = reasoning.reasoning_agent({"question": "labs", "doc_results": [], "db_results": [],
                                 "plan": {"missing": ["get_lab_results"], "use_documents": False}})
check("reasoning: missing patient ID asks for it", "patient ID" in out["answer"])
out = reasoning.reasoning_agent({"question": "?", "doc_results": [], "db_results": [],
                                 "plan": {"missing": [], "use_documents": True}})
check("reasoning: no evidence at all gives the 'no information' message", out["is_system_message"])
errored = {"plan": {"missing": []}, "db_results": [{"tool": "get_billing_summary", "arguments": {}, "status": "error", "message": "x"}]}
check("reasoning: notes list a failed tool", "get_billing_summary" in reasoning.access_notes(errored))

image_state = {"question": "summarise", "doc_results": [], "db_results": [], "image": saved,
               "plan": {"missing": [], "use_documents": False}}
evidence, sources = reasoning.build_evidence(image_state)
check("reasoning: uploaded image becomes an evidence block", "[Uploaded image: rx.png]" in evidence and sources[0]["type"] == "image")

# ---- 5. Evaluator ----
good = evaluator.JudgeScores(faithfulness=5, grounding=5, relevance=5, hallucination_detected=False,
                             adds_own_treatment=False, reason="ok")
ev_state = {"question": "labs", "answer": "Hemoglobin was 9.8 g/dL, flagged Low [Database: get_lab_results].",
            "sources": sources, "evidence": evidence + "\nHemoglobin 9.8 g/dL", "is_system_message": False,
            "doc_results": [], "db_results": [lab]}
evaluator.judge = lambda *a: good
out = evaluator.evaluator_agent(ev_state)
check("evaluator: a good answer is allowed with the disclaimer",
      out["allowed"] and "does not replace advice" in out["final_answer"])
check("evaluator: confidence is high and scores are reported",
      out["evaluation"]["confidence"] >= 0.9 and out["evaluation"]["faithfulness"] == 5)
check("evaluator: retrieval quality is reported", out["evaluation"]["retrieval_quality"] == 1.0)

bad_dose = {**ev_state, "answer": "Take 650 mg of paracetamol [Database: get_lab_results]."}
out = evaluator.evaluator_agent(bad_dose)
check("evaluator: a dose that is not in the evidence is blocked by the rules",
      not out["allowed"] and out["evaluation"]["decision"] == "blocked_by_rules")
out = evaluator.evaluator_agent({**ev_state, "answer": "Call 9876543210 for results [Database: get_lab_results]."})
check("evaluator: a phone number in the answer is masked", "9876543210" not in out["final_answer"])
for label, scores in [
    ("hallucination", evaluator.JudgeScores(faithfulness=4, grounding=4, relevance=4, hallucination_detected=True, adds_own_treatment=False)),
    ("low faithfulness", evaluator.JudgeScores(faithfulness=2, grounding=5, relevance=5, hallucination_detected=False, adds_own_treatment=False)),
    ("low grounding", evaluator.JudgeScores(faithfulness=5, grounding=2, relevance=5, hallucination_detected=False, adds_own_treatment=False)),
    ("own treatment", evaluator.JudgeScores(faithfulness=5, grounding=5, relevance=5, hallucination_detected=False, adds_own_treatment=True)),
]:
    evaluator.judge = lambda *a, s=scores: s
    out = evaluator.evaluator_agent(ev_state)
    check(f"evaluator: judge blocks {label}", not out["allowed"] and out["evaluation"]["decision"] == "blocked_by_judge")
evaluator.judge = lambda *a: evaluator.JudgeScores(faithfulness=5, grounding=5, relevance=5, hallucination_detected=False,
                                                   adds_own_treatment=False, in_medical_scope=False)
out = evaluator.evaluator_agent(ev_state)
check("evaluator: judge blocks an off-scope answer", not out["allowed"] and out["evaluation"]["decision"] == "blocked_off_scope")
evaluator.judge = lambda *a: evaluator.JudgeScores(faithfulness=3, grounding=3, relevance=3, hallucination_detected=False,
                                                   adds_own_treatment=False)
out = evaluator.evaluator_agent(ev_state)
check("evaluator: low confidence is allowed with a warning", out["allowed"] and "Confidence is low" in out["final_answer"])
evaluator.judge = boom
out = evaluator.evaluator_agent(ev_state)
check("evaluator: judge failure still applies rules and says so",
      out["allowed"] and out["evaluation"]["judge"] == "unavailable")
out = evaluator.evaluator_agent({"question": "x", "answer": "Please include the patient ID.", "sources": [],
                                 "evidence": "", "is_system_message": True, "doc_results": [], "db_results": []})
check("evaluator: fixed messages pass through unchanged", out["final_answer"] == "Please include the patient ID."
      and out["evaluation"]["decision"] == "no_answer")
try:
    evaluator.JudgeScores(faithfulness=9, grounding=1, relevance=1, hallucination_detected=False, adds_own_treatment=False)
    rejected = False
except ValueError:
    rejected = True
check("evaluator: judge scores outside 1 to 5 are rejected", rejected)

# ---- 6. All six agents in a row (fakes for everything outside) ----
planner.invoke_structured = lambda *a, **k: PLAN_FORM(use_database=True, use_image=True,
                                                      tools=["get_lab_results", "get_billing_summary"], patient_id="P00001")
mcp_agent.client.call_tool_sync = lambda n, a: lab["data"] if n == "get_lab_results" else {"count": 0, "bills": []}
vision_agent.get_extraction = lambda i: saved
retriever_agent.retrieve = lambda q: {"found": False, "results": []}
reasoning._ask = lambda q, e, n: "Hemoglobin was 9.8 g/dL [Database: get_lab_results]. The prescription lists Paracetamol [Uploaded image: rx.png]."
evaluator.judge = lambda *a: good
s = {"question": "Summarise this report and show labs for P00001", "extraction_id": "a" * 32}
for agent in (planner.planner_agent, retriever_agent.retriever_agent, mcp_agent.mcp_agent,
              vision_agent.vision_agent, reasoning.reasoning_agent, evaluator.evaluator_agent):
    s.update(agent(s))  # the same way LangGraph will merge each agent's output
check("pipeline: all six agents ran in order",
      [t["agent"] for t in s["trace"]] == ["planner", "retriever", "mcp", "vision", "reasoning", "evaluator"])
check("pipeline: final answer is allowed and uses database + image",
      s["allowed"] and {x["type"] for x in s["sources"]} == {"database", "image"})

passed = sum(1 for _, ok in results if ok)
print(f"\n{passed}/{len(results)} tests passed")
raise SystemExit(0 if passed == len(results) else 1)
