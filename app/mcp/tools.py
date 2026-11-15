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


TOOLS = [search_patients, get_patient_history, get_appointments, get_lab_results, get_billing_summary]
