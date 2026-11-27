"""
Read-only database tools. Each function returns a plain dictionary.

Rules used everywhere:
 - only the columns a tool needs are selected (never SELECT *)
 - phone, email and date of birth are never returned
 - every value goes in as a %s parameter
 - empty database values become "not recorded"
 - errors become {"error": "..."} so an agent never sees a stack trace
"""

import functools
import logging
import re

from app.mcp.connector import run_query

logger = logging.getLogger("mediassist.mcp")

NOT_RECORDED = "not recorded"
ID_PATTERN = re.compile(r"^P\d{1,10}$")  # real IDs look like P00001


def _clean_id(patient_id: str) -> str | None:
    """' p00001 ' -> 'P00001'. Returns None if it does not look like a patient ID."""
    value = (patient_id or "").strip().upper()
    return value if ID_PATTERN.match(value) else None


def _bad_id(patient_id: str) -> dict:
    return {"error": f"'{patient_id}' is not a valid patient ID. IDs look like P00123."}


def _fill(row: dict) -> dict:
    """Replace None and empty strings with 'not recorded'."""
    return {k: (NOT_RECORDED if v is None or str(v).strip() == "" else v) for k, v in row.items()}


def _to_number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _like(text: str) -> str:
    """Safe LIKE pattern: escape % and _ typed by the user."""
    escaped = text.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _safe(func):
    """Wrap a tool so any failure returns a short error instead of crashing."""
    @functools.wraps(func)  # keeps the real parameter names so MCP can read them
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Tool %s failed", func.__name__)
            return {"error": "The database could not be queried right now. Please try again."}
    return wrapper


def _limit(value: int, default: int, maximum: int) -> int:
    try:
        return max(1, min(int(value), maximum))
    except (TypeError, ValueError):
        return default


@_safe
def search_patients(name: str = "", city: str = "", patient_id: str = "", limit: int = 10) -> dict:
    """Find patients by name (part of first or last name), city, or patient ID."""
    conditions, params = [], []
    if patient_id.strip():
        pid = _clean_id(patient_id)
        if not pid:
            return _bad_id(patient_id)
        conditions.append("patient_id = %s")
        params.append(pid)
    if name.strip():
        conditions.append("(first_name || ' ' || last_name) ILIKE %s")
        params.append(_like(name))
    if city.strip():
        conditions.append("city ILIKE %s")
        params.append(_like(city))
    if not conditions:
        return {"error": "Give at least a name, a city or a patient ID."}
    rows = run_query(
        "SELECT patient_id, first_name, last_name, gender, blood_group, city, patient_status "
        "FROM patients WHERE " + " AND ".join(conditions) + " ORDER BY patient_id LIMIT %s",
        tuple(params) + (_limit(limit, 10, 50),),
    )
    return {"count": len(rows), "patients": [_fill(r) for r in rows]}


@_safe
def get_patient_history(patient_id: str) -> dict:
    """Patient profile with recent admissions and prescriptions."""
    pid = _clean_id(patient_id)
    if not pid:
        return _bad_id(patient_id)
    patient = run_query(
        "SELECT patient_id, first_name, last_name, gender, blood_group, city, state, "
        "registration_date, patient_status FROM patients WHERE patient_id = %s", (pid,))
    if not patient:
        return {"error": f"No patient with ID {pid}. IDs look like P00123."}
    admissions = run_query(
        "SELECT admission_id, admission_date, discharge_date, admission_type, diagnosis, "
        "admission_status, discharge_summary FROM admissions WHERE patient_id = %s "
        "ORDER BY admission_date DESC LIMIT 10", (pid,))
    prescriptions = run_query(
        "SELECT prescription_date, medicine_name, dosage, frequency, route, duration_days, "
        "instructions, prescription_status FROM prescriptions WHERE patient_id = %s "
        "ORDER BY prescription_date DESC LIMIT 10", (pid,))
    return {
        "patient": _fill(patient[0]),
        "admissions": [_fill(r) for r in admissions],
        "prescriptions": [_fill(r) for r in prescriptions],
    }


@_safe
def get_appointments(patient_id: str, status: str = "", upcoming_only: bool = False, limit: int = 10) -> dict:
    """Appointments of a patient, newest first. Optional status filter or upcoming only."""
    pid = _clean_id(patient_id)
    if not pid:
        return _bad_id(patient_id)
    sql_text = (
        "SELECT a.appointment_id, a.appointment_date, a.appointment_time, a.appointment_type, "
        "a.appointment_status, a.consultation_notes, d.doctor_name, d.specialization "
        "FROM appointments a LEFT JOIN doctors d ON d.doctor_id = a.doctor_id "
        "WHERE a.patient_id = %s"
    )
    params = [pid]
    if status.strip():
        sql_text += " AND a.appointment_status ILIKE %s"
        params.append(status.strip())
    if upcoming_only:  # dates are text in YYYY-MM-DD form, so text comparison works
        sql_text += " AND a.appointment_date >= to_char(CURRENT_DATE, 'YYYY-MM-DD')"
    sql_text += " ORDER BY a.appointment_date DESC, a.appointment_time DESC LIMIT %s"
    params.append(_limit(limit, 10, 50))
    rows = run_query(sql_text, tuple(params))
    return {"patient_id": pid, "count": len(rows), "appointments": [_fill(r) for r in rows]}


@_safe
def get_lab_results(patient_id: str, test_name: str = "", abnormal_only: bool = False, limit: int = 20) -> dict:
    """Lab results of a patient, newest first. Optional test name and abnormal-only filter."""
    pid = _clean_id(patient_id)
    if not pid:
        return _bad_id(patient_id)
    sql_text = (
        "SELECT test_name, test_category, test_date, result_value, result_unit, reference_min, "
        "reference_max, result_status, lab_name, report_notes FROM lab_results WHERE patient_id = %s"
    )
    params = [pid]
    if test_name.strip():
        sql_text += " AND test_name ILIKE %s"
        params.append(_like(test_name))
    if abnormal_only:
        sql_text += " AND result_status IS NOT NULL AND LOWER(result_status) <> 'normal'"
    sql_text += " ORDER BY test_date DESC LIMIT %s"
    params.append(_limit(limit, 20, 100))
    rows = run_query(sql_text, tuple(params))
    return {"patient_id": pid, "count": len(rows), "lab_results": [_fill(r) for r in rows]}


@_safe
def get_billing_summary(patient_id: str, limit: int = 10) -> dict:
    """Bills of a patient with totals, plus payments made against those bills."""
    pid = _clean_id(patient_id)
    if not pid:
        return _bad_id(patient_id)
    bills = run_query(
        "SELECT bill_id, bill_date, total_amount, insurance_covered, patient_payable, amount_paid, "
        "payment_status, payment_mode FROM billing WHERE patient_id = %s "
        "ORDER BY bill_date DESC LIMIT %s", (pid, _limit(limit, 10, 50)))
    payments = run_query(
        "SELECT p.bill_id, p.payment_date, p.amount, p.payment_mode, p.payment_status "
        "FROM payments p JOIN billing b ON b.bill_id = p.bill_id "
        "WHERE b.patient_id = %s ORDER BY p.payment_date DESC LIMIT 20", (pid,))
    totals = {  # added in Python because the column types are not guaranteed numeric
        "bills_shown": len(bills),
        "total_billed": round(sum(_to_number(b["total_amount"]) for b in bills), 2),
        "insurance_covered": round(sum(_to_number(b["insurance_covered"]) for b in bills), 2),
        "patient_payable": round(sum(_to_number(b["patient_payable"]) for b in bills), 2),
        "amount_paid": round(sum(_to_number(b["amount_paid"]) for b in bills), 2),
    }
    return {"patient_id": pid, "totals": totals,
            "bills": [_fill(r) for r in bills], "payments": [_fill(r) for r in payments]}


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # dates are stored as text like 2026-06-08
YEAR_PATTERN = re.compile(r"^\d{4}$")


@_safe
def get_patient_summary(patient_id: str) -> dict:
    """One-page overview of a patient: profile, insurer, counts of records, latest admission, billing totals."""
    pid = _clean_id(patient_id)
    if not pid:
        return _bad_id(patient_id)
    patient = run_query(
        "SELECT p.patient_id, p.first_name, p.last_name, p.gender, p.blood_group, p.city, p.state, "
        "p.registration_date, p.patient_status, i.provider_name AS insurer, i.policy_type "
        "FROM patients p LEFT JOIN insurance_policies i ON i.insurance_id = p.insurance_id "
        "WHERE p.patient_id = %s", (pid,))
    if not patient:
        return {"error": f"No patient with ID {pid}. IDs look like P00123."}
    counts = run_query(
        "SELECT (SELECT COUNT(*) FROM admissions WHERE patient_id = %s) AS admissions, "
        "(SELECT COUNT(*) FROM appointments WHERE patient_id = %s) AS appointments, "
        "(SELECT COUNT(*) FROM prescriptions WHERE patient_id = %s) AS prescriptions, "
        "(SELECT COUNT(*) FROM lab_results WHERE patient_id = %s) AS lab_results, "
        "(SELECT COUNT(*) FROM lab_results WHERE patient_id = %s AND result_status IS NOT NULL "
        " AND LOWER(result_status) <> 'normal') AS abnormal_lab_results", (pid,) * 5)
    latest = run_query(
        "SELECT a.admission_date, a.discharge_date, a.diagnosis, a.admission_status, "
        "d.department_name, doc.doctor_name FROM admissions a "
        "LEFT JOIN departments d ON d.department_id = a.department_id "
        "LEFT JOIN doctors doc ON doc.doctor_id = a.doctor_id "
        "WHERE a.patient_id = %s ORDER BY a.admission_date DESC LIMIT 1", (pid,))
    money = run_query(
        "SELECT COUNT(*) AS bills, COALESCE(SUM(total_amount), 0) AS total_billed, "
        "COALESCE(SUM(amount_paid), 0) AS amount_paid FROM billing WHERE patient_id = %s", (pid,))
    return {
        "patient": _fill(patient[0]),
        "record_counts": counts[0],
        "latest_admission": _fill(latest[0]) if latest else {"info": "no admissions"},
        "billing_totals": {k: (round(_to_number(v), 2) if k != "bills" else v) for k, v in money[0].items()},
    }


@_safe
def get_prescriptions(patient_id: str, medicine: str = "", status: str = "", limit: int = 20) -> dict:
    """Medicines prescribed to a patient, newest first. Optional medicine name and status filter."""
    pid = _clean_id(patient_id)
    if not pid:
        return _bad_id(patient_id)
    sql_text = (
        "SELECT p.prescription_date, p.medicine_name, p.dosage, p.frequency, p.route, p.duration_days, "
        "p.instructions, p.prescription_status, d.doctor_name "
        "FROM prescriptions p LEFT JOIN doctors d ON d.doctor_id = p.doctor_id WHERE p.patient_id = %s"
    )
    params = [pid]
    if medicine.strip():
        sql_text += " AND p.medicine_name ILIKE %s"
        params.append(_like(medicine))
    if status.strip():
        sql_text += " AND p.prescription_status ILIKE %s"
        params.append(status.strip())
    sql_text += " ORDER BY p.prescription_date DESC LIMIT %s"
    params.append(_limit(limit, 20, 100))
    rows = run_query(sql_text, tuple(params))
    return {"patient_id": pid, "count": len(rows), "prescriptions": [_fill(r) for r in rows]}


@_safe
def get_admissions(patient_id: str, status: str = "", limit: int = 10) -> dict:
    """Hospital admissions of a patient, newest first, with department, doctor and branch."""
    pid = _clean_id(patient_id)
    if not pid:
        return _bad_id(patient_id)
    sql_text = (
        "SELECT a.admission_id, a.admission_date, a.discharge_date, a.admission_type, a.diagnosis, "
        "a.admission_status, a.discharge_summary, d.department_name, doc.doctor_name, b.branch_name "
        "FROM admissions a LEFT JOIN departments d ON d.department_id = a.department_id "
        "LEFT JOIN doctors doc ON doc.doctor_id = a.doctor_id "
        "LEFT JOIN hospital_branches b ON b.branch_id = a.branch_id WHERE a.patient_id = %s"
    )
    params = [pid]
    if status.strip():
        sql_text += " AND a.admission_status ILIKE %s"
        params.append(status.strip())
    sql_text += " ORDER BY a.admission_date DESC LIMIT %s"
    params.append(_limit(limit, 10, 50))
    rows = run_query(sql_text, tuple(params))
    return {"patient_id": pid, "count": len(rows), "admissions": [_fill(r) for r in rows]}


@_safe
def get_doctor_schedule(doctor_name: str = "", doctor_id: str = "", date: str = "", limit: int = 20) -> dict:
    """Appointments of a doctor (by name or doctor ID) on one date (YYYY-MM-DD), or all upcoming if no date."""
    if not doctor_name.strip() and not doctor_id.strip():
        return {"error": "Give a doctor name or a doctor ID."}
    sql_text = (
        "SELECT d.doctor_id, d.doctor_name, d.specialization, a.appointment_date, a.appointment_time, "
        "a.appointment_type, a.appointment_status, a.patient_id "
        "FROM appointments a JOIN doctors d ON d.doctor_id = a.doctor_id WHERE 1 = 1"
    )
    params = []
    if doctor_id.strip():
        sql_text += " AND d.doctor_id = %s"
        params.append(doctor_id.strip().upper())
    if doctor_name.strip():
        sql_text += " AND d.doctor_name ILIKE %s"
        params.append(_like(doctor_name))
    if date.strip():
        if not DATE_PATTERN.match(date.strip()):
            return {"error": "Dates must look like 2026-11-27 (year-month-day)."}
        sql_text += " AND a.appointment_date = %s"
        params.append(date.strip())
    else:
        sql_text += " AND a.appointment_date >= to_char(CURRENT_DATE, 'YYYY-MM-DD')"
    sql_text += " ORDER BY a.appointment_date, a.appointment_time LIMIT %s"
    params.append(_limit(limit, 20, 100))
    rows = run_query(sql_text, tuple(params))
    return {"count": len(rows), "schedule": [_fill(r) for r in rows]}


@_safe
def search_doctors(name: str = "", specialization: str = "", department: str = "", city: str = "",
                   limit: int = 10) -> dict:
    """Find doctors by name, specialization, department or branch city."""
    conditions, params = [], []
    if name.strip():
        conditions.append("d.doctor_name ILIKE %s")
        params.append(_like(name))
    if specialization.strip():
        conditions.append("d.specialization ILIKE %s")
        params.append(_like(specialization))
    if department.strip():
        conditions.append("dep.department_name ILIKE %s")
        params.append(_like(department))
    if city.strip():
        conditions.append("b.city ILIKE %s")
        params.append(_like(city))
    if not conditions:
        return {"error": "Give at least a name, specialization, department or city."}
    rows = run_query(
        "SELECT d.doctor_id, d.doctor_name, d.specialization, dep.department_name, b.branch_name, b.city "
        "FROM doctors d LEFT JOIN departments dep ON dep.department_id = d.department_id "
        "LEFT JOIN hospital_branches b ON b.branch_id = d.branch_id WHERE " + " AND ".join(conditions) +
        " ORDER BY d.doctor_name LIMIT %s", tuple(params) + (_limit(limit, 10, 50),))
    return {"count": len(rows), "doctors": [_fill(r) for r in rows]}


@_safe
def search_lab_tests(test_name: str = "", status: str = "", category: str = "", limit: int = 20) -> dict:
    """Find lab results across all patients by test name, status (e.g. High, Low) or category, with counts per status."""
    conditions, params = [], []
    if test_name.strip():
        conditions.append("test_name ILIKE %s")
        params.append(_like(test_name))
    if status.strip():
        conditions.append("result_status ILIKE %s")
        params.append(status.strip())
    if category.strip():
        conditions.append("test_category ILIKE %s")
        params.append(_like(category))
    if not conditions:
        return {"error": "Give at least a test name, a status or a category."}
    where = " AND ".join(conditions)
    by_status = run_query(
        "SELECT COALESCE(result_status, 'not recorded') AS result_status, COUNT(*) AS total "
        "FROM lab_results WHERE " + where + " GROUP BY 1 ORDER BY total DESC", tuple(params))
    rows = run_query(
        "SELECT patient_id, test_name, test_category, test_date, result_value, result_unit, result_status "
        "FROM lab_results WHERE " + where + " ORDER BY test_date DESC LIMIT %s",
        tuple(params) + (_limit(limit, 20, 100),))
    return {"count_by_status": [_fill(r) for r in by_status], "shown": len(rows),
            "matches": [_fill(r) for r in rows]}


@_safe
def get_insurance_details(patient_id: str = "", provider: str = "", limit: int = 10) -> dict:
    """A patient's insurance policy and claims, OR (no patient ID) billing totals per insurance provider."""
    if patient_id.strip():
        pid = _clean_id(patient_id)
        if not pid:
            return _bad_id(patient_id)
        policy = run_query(
            "SELECT i.insurance_id, i.provider_name, i.policy_type, i.coverage_limit "
            "FROM patients p LEFT JOIN insurance_policies i ON i.insurance_id = p.insurance_id "
            "WHERE p.patient_id = %s", (pid,))
        if not policy:
            return {"error": f"No patient with ID {pid}. IDs look like P00123."}
        bills = run_query(
            "SELECT COUNT(*) AS bills, COALESCE(SUM(total_amount), 0) AS total_billed, "
            "COALESCE(SUM(insurance_covered), 0) AS insurance_covered, "
            "COALESCE(SUM(patient_payable), 0) AS patient_payable FROM billing WHERE patient_id = %s", (pid,))
        return {"patient_id": pid, "policy": _fill(policy[0]),
                "billing_totals": {k: (v if k == "bills" else round(_to_number(v), 2)) for k, v in bills[0].items()}}
    sql_text = (
        "SELECT i.provider_name, COUNT(DISTINCT i.insurance_id) AS policies, COUNT(b.bill_id) AS bills, "
        "COALESCE(SUM(b.total_amount), 0) AS total_billed, COALESCE(SUM(b.insurance_covered), 0) AS insurance_covered, "
        "COALESCE(SUM(b.patient_payable), 0) AS patient_payable "
        "FROM insurance_policies i LEFT JOIN billing b ON b.insurance_id = i.insurance_id"
    )
    params = []
    if provider.strip():
        sql_text += " WHERE i.provider_name ILIKE %s"
        params.append(_like(provider))
    sql_text += " GROUP BY i.provider_name ORDER BY total_billed DESC LIMIT %s"
    params.append(_limit(limit, 10, 50))
    rows = run_query(sql_text, tuple(params))
    out = [{k: (round(_to_number(v), 2) if k in ("total_billed", "insurance_covered", "patient_payable") else v)
            for k, v in r.items()} for r in rows]
    return {"count": len(out), "by_provider": out}


METRICS = ("admissions_by_department", "appointments_by_branch", "top_diagnoses", "billing_by_department")


@_safe
def get_department_statistics(metric: str = "admissions_by_department", year: str = "", city: str = "",
                              department: str = "", limit: int = 10) -> dict:
    """Hospital-wide statistics. metric = admissions_by_department | appointments_by_branch |
    top_diagnoses | billing_by_department. Optional year (2025), city and department filters."""
    metric = metric.strip().lower()
    if metric not in METRICS:
        return {"error": "Unknown metric. Use one of: " + ", ".join(METRICS) + "."}
    year = str(year).strip()
    if year and not YEAR_PATTERN.match(year):
        return {"error": "Year must have 4 digits, for example 2025."}
    n = _limit(limit, 10, 50)
    params: list = []

    def where(date_col: str, with_city: str = "", with_dept: str = "") -> str:
        parts = []
        if year:
            parts.append(f"LEFT({date_col}, 4) = %s")
            params.append(year)
        if city.strip() and with_city:
            parts.append(f"{with_city} ILIKE %s")
            params.append(_like(city))
        if department.strip() and with_dept:
            parts.append(f"{with_dept} ILIKE %s")
            params.append(_like(department))
        return (" WHERE " + " AND ".join(parts)) if parts else ""

    if metric == "admissions_by_department":
        sql_text = ("SELECT d.department_name, COUNT(*) AS admissions FROM admissions a "
                    "JOIN departments d ON d.department_id = a.department_id "
                    "LEFT JOIN hospital_branches b ON b.branch_id = a.branch_id"
                    + where("a.admission_date", "b.city", "d.department_name")
                    + " GROUP BY d.department_name ORDER BY admissions DESC LIMIT %s")
    elif metric == "appointments_by_branch":
        sql_text = ("SELECT b.branch_name, LEFT(a.appointment_date, 7) AS month, COUNT(*) AS appointments "
                    "FROM appointments a JOIN hospital_branches b ON b.branch_id = a.branch_id"
                    + where("a.appointment_date", "b.city")
                    + " GROUP BY b.branch_name, month ORDER BY month DESC, appointments DESC LIMIT %s")
        n = _limit(limit, 24, 100)
    elif metric == "top_diagnoses":
        sql_text = ("SELECT a.diagnosis, COUNT(*) AS cases FROM admissions a "
                    "JOIN patients p ON p.patient_id = a.patient_id "
                    "LEFT JOIN departments d ON d.department_id = a.department_id"
                    + where("a.admission_date", "p.city", "d.department_name")
                    + (" AND" if (year or city.strip() or department.strip()) else " WHERE")
                    + " a.diagnosis IS NOT NULL GROUP BY a.diagnosis ORDER BY cases DESC LIMIT %s")
    else:  # billing_by_department
        sql_text = ("SELECT d.department_name, COUNT(*) AS bills, COALESCE(SUM(b.total_amount), 0) AS total_billed "
                    "FROM billing b JOIN admissions a ON a.admission_id = b.admission_id "
                    "JOIN departments d ON d.department_id = a.department_id"
                    + where("b.bill_date", "", "d.department_name")
                    + " GROUP BY d.department_name ORDER BY total_billed DESC LIMIT %s")
    rows = run_query(sql_text, tuple(params) + (n,))
    rows = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
    return {"metric": metric, "year": year or "all years", "count": len(rows), "rows": rows}


TOOLS = [search_patients, get_patient_history, get_appointments, get_lab_results, get_billing_summary,
         get_patient_summary, get_prescriptions, get_admissions, get_doctor_schedule, search_doctors,
         search_lab_tests, get_insurance_details, get_department_statistics]
