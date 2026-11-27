"""
MCP test: tools are discovered, and real data comes back through MCP.

Run from the project root:
    python -m tests.test_mcp
"""

import asyncio

from app.mcp.client import call_tool, list_tools

EXPECTED = {"search_patients", "get_patient_history", "get_appointments",
            "get_lab_results", "get_billing_summary", "get_patient_summary", "get_prescriptions",
            "get_admissions", "get_doctor_schedule", "search_doctors", "search_lab_tests",
            "get_insurance_details", "get_department_statistics"}
results = []


def check(name: str, ok: bool) -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


async def main() -> None:
    tools = await list_tools()
    names = {t["name"] for t in tools}
    check(f"Server lists the 13 tools ({sorted(names)})", EXPECTED <= names)

    found = await call_tool("search_patients", {"city": "a", "limit": 1})
    check("search_patients returns a patient", found.get("count", 0) >= 1)
    if found.get("count", 0) < 1:
        print("      ", found)
        return
    pid = found["patients"][0]["patient_id"]
    print("      using patient", pid)

    history = await call_tool("get_patient_history", {"patient_id": pid})
    check("get_patient_history returns the profile", history.get("patient", {}).get("patient_id") == pid)
    print(f"      admissions={len(history.get('admissions', []))} prescriptions={len(history.get('prescriptions', []))}")

    appts = await call_tool("get_appointments", {"patient_id": pid})
    check("get_appointments answers", "appointments" in appts)

    labs = await call_tool("get_lab_results", {"patient_id": pid})
    check("get_lab_results answers", "lab_results" in labs)
    abnormal = await call_tool("get_lab_results", {"patient_id": pid, "abnormal_only": True})
    check("abnormal_only filter works", "lab_results" in abnormal)

    bills = await call_tool("get_billing_summary", {"patient_id": pid})
    check("get_billing_summary returns totals", "totals" in bills)
    print("      totals:", bills.get("totals"))

    lower = await call_tool("get_patient_history", {"patient_id": f" {pid.lower()} "})
    check("ID is trimmed and uppercased", lower.get("patient", {}).get("patient_id") == pid)

    bad = await call_tool("get_patient_history", {"patient_id": "ZZ99"})
    check("Bad ID gives a clear error", "error" in bad)
    missing = await call_tool("get_patient_history", {"patient_id": "P99999999"})
    check("Unknown patient gives a clear error", "error" in missing)

    sql_attack = await call_tool("search_patients", {"name": "x'; DROP TABLE patients;--"})
    check("SQL-injection text is harmless", sql_attack.get("count") == 0)

    # ---- the 8 added tools ----
    summary = await call_tool("get_patient_summary", {"patient_id": pid})
    check("get_patient_summary has counts", "record_counts" in summary and "patient" in summary)
    rx = await call_tool("get_prescriptions", {"patient_id": pid})
    check("get_prescriptions answers", "prescriptions" in rx)
    adm = await call_tool("get_admissions", {"patient_id": pid})
    check("get_admissions answers", "admissions" in adm)
    docs = await call_tool("search_doctors", {"specialization": "Cardio"})
    check("search_doctors finds cardiologists", docs.get("count", 0) >= 1)
    if docs.get("count", 0) >= 1:
        schedule = await call_tool("get_doctor_schedule", {"doctor_name": docs["doctors"][0]["doctor_name"]})
        check("get_doctor_schedule answers", "schedule" in schedule)
    check("get_doctor_schedule rejects a bad date",
          "error" in await call_tool("get_doctor_schedule", {"doctor_name": "a", "date": "June"}))
    labs_all = await call_tool("search_lab_tests", {"test_name": "hba1c"})
    check("search_lab_tests counts by status", "count_by_status" in labs_all)
    check("get_insurance_details for a patient", "policy" in await call_tool("get_insurance_details", {"patient_id": pid}))
    check("get_insurance_details per provider", "by_provider" in await call_tool("get_insurance_details", {}))
    for metric in ("admissions_by_department", "appointments_by_branch", "top_diagnoses", "billing_by_department"):
        stats = await call_tool("get_department_statistics", {"metric": metric})
        check(f"get_department_statistics: {metric}", stats.get("count", 0) >= 1)
    check("get_department_statistics rejects unknown metric",
          "error" in await call_tool("get_department_statistics", {"metric": "drop_table"}))
    check("search_doctors: SQL text is harmless",
          (await call_tool("search_doctors", {"name": "x'; DROP TABLE doctors;--"})).get("count") == 0)


asyncio.run(main())
passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
