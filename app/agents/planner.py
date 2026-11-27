"""
Planner Agent: reads the question and decides which sources are needed.

It only plans, it never answers. The LLM fills a small plan form. Then plain code
checks every field (tool names, patient ID) so a wrong LLM output cannot cause harm.
If the LLM is down, simple keyword rules make the plan instead.

Plan = {"use_documents", "use_database", "use_image", "calls": [{"name", "arguments"}],
        "missing": [...], "subtasks": [...], "source": "llm" or "rules"}
"""

import logging
import re
import time

from pydantic import BaseModel, Field

from app import prompts
from app.agents.state import AgentState, add_trace
from app.guardrails.input_guard import ALLOWED_TOOLS
from app.llm import invoke_structured

logger = logging.getLogger("mediassist.agents")

PATIENT_ID = re.compile(r"\bP\d{3,10}\b", re.IGNORECASE)
NEEDS_PATIENT_ID = {"get_patient_history", "get_appointments", "get_lab_results", "get_billing_summary",
                    "get_patient_summary", "get_prescriptions", "get_admissions"}
STATS_METRICS = ("admissions_by_department", "appointments_by_branch", "top_diagnoses", "billing_by_department")
YEAR = re.compile(r"\b(20\d{2})\b")
ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

TOOL_WORDS = [  # keyword rules, used only when the LLM is unavailable
    ("get_department_statistics", r"which department|department.*(most|highest|busiest)|trends?|top diagnos|most common diagnos|by branch|across branches|statistics"),
    ("search_doctors", r"find (?:a )?doctors?|doctors? (?:in|for|available)|specialists?|who is dr|list (?:of )?doctors"),
    ("get_doctor_schedule", r"doctor.{0,20}schedule|dr\.? \w+.{0,20}(?:schedule|appointments)"),
    ("search_lab_tests", r"(?:all|any) patients.{0,30}(?:test|lab)|abnormal .{0,15}across|how many.{0,30}(?:test|lab)"),
    ("get_insurance_details", r"insurance (?:provider|details|policy)|by insur|insurer"),
    ("get_prescriptions", r"prescri|medicines?|medications?"),
    ("get_admissions", r"admission|admitted|discharge"),
    ("get_patient_summary", r"summary|overview"),
    ("get_lab_results", r"\blabs?\b|test result|hba1c|blood|hemoglobin|glucose|abnormal|pathology"),
    ("get_appointments", r"appointment|schedule|follow[- ]?up|upcoming|visit"),
    ("get_billing_summary", r"\bbill|payment|\bpaid\b|invoice|claim|insurance|outstanding"),
    ("get_patient_history", r"history|diagnos"),
]
DOCUMENT_WORDS = re.compile(r"policy|process|procedure|\bsop\b|guideline|workflow|escalat|compliance|"
                            r"refund|\bsteps?\b|how (?:do|to|can)|what is|documents?|rules?", re.IGNORECASE)
NAME_IN_TEXT = re.compile(r"(?:named|called|patient)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")


class Plan(BaseModel):  # the form the LLM fills in (flat fields work best with small models)
    use_documents: bool = Field(default=False, description="Hospital documents are needed")
    use_database: bool = Field(default=False, description="Patient records are needed")
    use_image: bool = Field(default=False, description="The uploaded image is needed")
    tools: list[str] = Field(default_factory=list, description="Database tool names")
    patient_id: str = Field(default="", description="Patient ID like P00123, or empty")
    search_name: str = Field(default="", description="Name for search_patients")
    search_city: str = Field(default="", description="City for search_patients, search_doctors or statistics")
    doctor_name: str = Field(default="", description="Doctor name for doctor tools")
    specialization: str = Field(default="", description="Doctor specialization or department name")
    metric: str = Field(default="", description="For get_department_statistics: admissions_by_department, appointments_by_branch, top_diagnoses or billing_by_department")
    year: str = Field(default="", description="4-digit year like 2025, or empty")
    date: str = Field(default="", description="Date as YYYY-MM-DD, or empty")
    medicine: str = Field(default="", description="Medicine name filter")
    insurer: str = Field(default="", description="Insurance provider name filter")
    lab_status: str = Field(default="", description="Lab status such as High, Low, Normal, Critical")
    test_name: str = Field(default="", description="Lab test name filter")
    abnormal_only: bool = Field(default=False, description="Only abnormal lab results")
    upcoming_only: bool = Field(default=False, description="Only upcoming appointments")
    subtasks: list[str] = Field(default_factory=list, description="1 to 4 short steps")


def _id_in(text: str) -> str:
    match = PATIENT_ID.search(text or "")
    return match.group(0).upper() if match else ""


def rule_plan(question: str, has_image: bool, image_patient_id: str = "") -> Plan:
    """Keyword plan, used when the LLM fails."""
    pid = _id_in(question) or image_patient_id
    tools = [name for name, words in TOOL_WORDS if re.search(words, question, re.IGNORECASE)]
    if not pid and any(t not in NEEDS_PATIENT_ID for t in tools):  # a hospital-wide question
        tools = [t for t in tools if t not in NEEDS_PATIENT_ID]
    name = NAME_IN_TEXT.search(question)
    if name and not pid:
        tools = ["search_patients"]
    if pid and not tools and not has_image:
        tools = ["get_patient_history"]
    metric = ""
    if "get_department_statistics" in tools:
        low = question.lower()
        metric = ("appointments_by_branch" if re.search(r"appointment|branch", low)
                  else "top_diagnoses" if "diagnos" in low
                  else "billing_by_department" if re.search(r"bill|revenue", low) else "admissions_by_department")
    year = YEAR.search(question)
    use_documents = bool(DOCUMENT_WORDS.search(question)) or not (pid or tools or has_image)
    return Plan(
        use_documents=use_documents, use_database=bool(tools), use_image=has_image, tools=tools,
        patient_id=pid, search_name=name.group(1) if name else "", metric=metric,
        year=year.group(1) if year else "", date=(ISO_DATE.search(question) or [None, ""])[1],
        abnormal_only=bool(re.search(r"abnormal", question, re.IGNORECASE)),
        upcoming_only=bool(re.search(r"upcoming|next|future", question, re.IGNORECASE)),
        subtasks=["Collect the needed records", "Write a grounded answer"])


def build_plan(question: str, plan: Plan, has_image: bool, image_patient_id: str, source: str) -> dict:
    """Check the plan and turn it into tool calls. Nothing from the LLM is trusted as it is."""
    pid = _id_in(question) or _id_in(plan.patient_id) or image_patient_id  # the question wins
    tools = [t for t in dict.fromkeys(plan.tools) if t in ALLOWED_TOOLS]  # drop unknown, drop duplicates
    if pid and plan.use_database and not tools:
        tools = ["get_patient_history"]

    calls, missing = [], []
    year = plan.year.strip() if re.fullmatch(r"\d{4}", plan.year.strip()) else ""
    date = plan.date.strip() if re.fullmatch(r"\d{4}-\d{2}-\d{2}", plan.date.strip()) else ""
    text = lambda v: v.strip()[:60]  # every free-text argument is short

    def add(tool: str, args: dict) -> None:
        calls.append({"name": tool, "arguments": {k: v for k, v in args.items() if v not in ("", None, False)}})

    for tool in tools:
        if tool in NEEDS_PATIENT_ID and not pid:
            missing.append(tool)  # this tool needs a patient ID and we have none
        elif tool == "search_patients":
            args = {"name": text(plan.search_name), "city": text(plan.search_city), "patient_id": pid}
            if any(args.values()):
                add(tool, args)
            else:
                missing.append(tool)
        elif tool == "get_lab_results":
            add(tool, {"patient_id": pid, "test_name": text(plan.test_name), "abnormal_only": plan.abnormal_only})
        elif tool == "get_appointments":
            add(tool, {"patient_id": pid, "upcoming_only": plan.upcoming_only})
        elif tool == "get_prescriptions":
            add(tool, {"patient_id": pid, "medicine": text(plan.medicine)})
        elif tool in ("get_patient_history", "get_billing_summary", "get_patient_summary", "get_admissions"):
            add(tool, {"patient_id": pid})
        elif tool == "get_doctor_schedule":
            if text(plan.doctor_name):
                add(tool, {"doctor_name": text(plan.doctor_name), "date": date})
            else:
                missing.append(tool)
        elif tool == "search_doctors":
            args = {"name": text(plan.doctor_name), "specialization": text(plan.specialization),
                    "city": text(plan.search_city)}
            if any(args.values()):
                add(tool, args)
            else:
                missing.append(tool)
        elif tool == "search_lab_tests":
            args = {"test_name": text(plan.test_name), "status": text(plan.lab_status)}
            if any(args.values()):
                add(tool, args)
            else:
                missing.append(tool)
        elif tool == "get_insurance_details":
            add(tool, {"patient_id": pid, "provider": text(plan.insurer)})
        elif tool == "get_department_statistics":
            metric = plan.metric.strip().lower()
            add(tool, {"metric": metric if metric in STATS_METRICS else "admissions_by_department",
                       "year": year, "city": text(plan.search_city), "department": text(plan.specialization)})

    use_documents = plan.use_documents or not (calls or missing or has_image)  # never plan "nothing"
    return {
        "use_documents": use_documents, "use_database": bool(calls), "use_image": has_image,  # an attached image is always used
        "calls": calls, "missing": missing, "patient_id": pid,
        "subtasks": [s[:120] for s in plan.subtasks[:4]], "source": source,
    }


def planner_agent(state: AgentState) -> dict:
    started = time.time()
    question = state["question"]
    image = state.get("image") or {}
    has_image = bool(state.get("extraction_id") or image)
    image_pid = _id_in(image.get("ocr_text", ""))  # "Patient ID: P00123" read from the image

    try:
        plan = invoke_structured(prompts.PLANNER_PROMPT, Plan, {
            "question": question, "has_image": "yes" if has_image else "no",
            "image_patient_id": image_pid or "(none)"})
        result = build_plan(question, plan, has_image, image_pid, "llm")
    except Exception:
        logger.exception("Planner LLM failed; using keyword rules")
        result = build_plan(question, rule_plan(question, has_image, image_pid), has_image, image_pid, "rules")

    note = (f"source={result['source']} docs={result['use_documents']} "
            f"tools={[c['name'] for c in result['calls']]} image={result['use_image']}")
    return {"plan": result, "trace": add_trace(state, "planner", started, note)}
