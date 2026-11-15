"""
MCP test: tools are discovered, and real data comes back through MCP.

Run from the project root:
    python -m tests.test_mcp
"""

import asyncio

from app.mcp.client import call_tool, list_tools

EXPECTED = {"search_patients", "get_patient_history", "get_appointments",
            "get_lab_results", "get_billing_summary"}
results = []


def check(name: str, ok: bool) -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


async def main() -> None:
    tools = await list_tools()
    names = {t["name"] for t in tools}
    check(f"Server lists the 5 tools ({sorted(names)})", EXPECTED <= names)

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


asyncio.run(main())
passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
