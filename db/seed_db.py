"""
Create the MediAssist schema and fill it with SYNTHETIC (fake) hospital data.

Usage (reads DB_* settings from .env, same as the app):
    python db/seed_db.py              # default size
    python db/seed_db.py --patients 5000

No real patient data is used. Names, IDs and values are randomly generated.
"""
import argparse
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

random.seed(42)  # same data every run

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Rahul", "Amit", "Sanjay", "Vikram",
         "Ananya", "Priya", "Neha", "Pooja", "Sneha", "Kavya", "Isha", "Meera", "Divya", "Riya",
         "Suresh", "Ramesh", "Mahesh", "Deepak", "Nikhil", "Anjali", "Sunita", "Kiran", "Shweta", "Tanvi"]
LAST = ["Sharma", "Patil", "Deshmukh", "Kulkarni", "Joshi", "Iyer", "Reddy", "Nair", "Singh", "Gupta",
        "Mehta", "Shah", "Verma", "Khan", "Das", "Bose", "Rao", "Jadhav", "Pawar", "More"]
CITIES = [("Pune", "Maharashtra"), ("Mumbai", "Maharashtra"), ("Nagpur", "Maharashtra"),
          ("Bengaluru", "Karnataka"), ("Hyderabad", "Telangana"), ("Chennai", "Tamil Nadu"),
          ("Delhi", "Delhi"), ("Kolkata", "West Bengal"), ("Ahmedabad", "Gujarat"), ("Jaipur", "Rajasthan")]
BLOOD = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
DEPARTMENTS = ["Cardiology", "Endocrinology", "General Medicine", "Orthopedics", "Neurology",
               "Pediatrics", "Gynecology", "Nephrology", "Pulmonology", "Oncology",
               "Emergency", "Gastroenterology"]
DEPT_SPEC = {d: d for d in DEPARTMENTS}
INSURERS = ["Star Health", "HDFC Ergo", "ICICI Lombard", "Niva Bupa", "Care Health",
            "Bajaj Allianz", "New India Assurance", "Max Bupa"]
DIAGNOSES = {  # diagnosis -> (department, medicines [(name, dosage, frequency, route)])
    "Type 2 Diabetes Mellitus": ("Endocrinology", [("Metformin", "500 mg", "Twice daily", "Oral"), ("Glimepiride", "1 mg", "Once daily", "Oral")]),
    "Hypertension": ("Cardiology", [("Amlodipine", "5 mg", "Once daily", "Oral"), ("Telmisartan", "40 mg", "Once daily", "Oral")]),
    "Acute Myocardial Infarction": ("Cardiology", [("Aspirin", "75 mg", "Once daily", "Oral"), ("Atorvastatin", "40 mg", "Once daily at night", "Oral")]),
    "Community Acquired Pneumonia": ("Pulmonology", [("Amoxicillin-Clavulanate", "625 mg", "Three times daily", "Oral"), ("Paracetamol", "500 mg", "As needed", "Oral")]),
    "COPD Exacerbation": ("Pulmonology", [("Salbutamol", "2 puffs", "Four times daily", "Inhaled"), ("Prednisolone", "30 mg", "Once daily", "Oral")]),
    "Chronic Kidney Disease": ("Nephrology", [("Furosemide", "40 mg", "Once daily", "Oral"), ("Calcium Acetate", "667 mg", "Three times daily", "Oral")]),
    "Fracture of Femur": ("Orthopedics", [("Diclofenac", "50 mg", "Twice daily", "Oral"), ("Calcium + Vitamin D3", "500 mg", "Once daily", "Oral")]),
    "Ischemic Stroke": ("Neurology", [("Clopidogrel", "75 mg", "Once daily", "Oral"), ("Atorvastatin", "40 mg", "Once daily at night", "Oral")]),
    "Gastroenteritis": ("Gastroenterology", [("Ondansetron", "4 mg", "Twice daily", "Oral"), ("ORS", "1 sachet", "After each loose stool", "Oral")]),
    "Bronchial Asthma": ("Pulmonology", [("Budesonide", "200 mcg", "Twice daily", "Inhaled"), ("Montelukast", "10 mg", "Once daily at night", "Oral")]),
    "Iron Deficiency Anemia": ("General Medicine", [("Ferrous Sulfate", "200 mg", "Twice daily", "Oral"), ("Folic Acid", "5 mg", "Once daily", "Oral")]),
    "Urinary Tract Infection": ("General Medicine", [("Nitrofurantoin", "100 mg", "Twice daily", "Oral")]),
    "Typhoid Fever": ("General Medicine", [("Ceftriaxone", "1 g", "Once daily", "Intravenous")]),
    "Normal Delivery": ("Gynecology", [("Ferrous Sulfate", "100 mg", "Once daily", "Oral")]),
    "Childhood Fever": ("Pediatrics", [("Paracetamol", "250 mg", "Every 6 hours", "Oral")]),
}
# test_name, category, unit, ref_min, ref_max
LABS = [("HbA1c", "Diabetes", "%", 4.0, 5.6), ("Fasting Blood Sugar", "Diabetes", "mg/dL", 70, 100),
        ("Hemoglobin", "Hematology", "g/dL", 12.0, 16.0), ("Creatinine", "Renal", "mg/dL", 0.6, 1.3),
        ("Total Cholesterol", "Lipid", "mg/dL", 0, 200), ("LDL Cholesterol", "Lipid", "mg/dL", 0, 100),
        ("TSH", "Thyroid", "mIU/L", 0.4, 4.0), ("WBC Count", "Hematology", "10^3/uL", 4.0, 11.0),
        ("Platelet Count", "Hematology", "10^3/uL", 150, 400), ("Blood Urea", "Renal", "mg/dL", 15, 40)]
LAB_NAMES = ["Apollo Diagnostics", "Metropolis Labs", "Thyrocare", "In-house Lab"]
ADM_TYPES = ["Emergency", "Planned", "Transfer"]
APPT_TYPES = ["Consultation", "Follow-up", "Routine Check-up", "Procedure"]
APPT_STATUS = ["Completed", "Completed", "Completed", "Scheduled", "Cancelled", "No-show"]
PAY_MODES = ["Cash", "Card", "UPI", "Insurance", "Net Banking"]
TODAY = date(2026, 11, 27)


def iso(d):
    return d.isoformat()


def rand_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))


def pid(n):
    return f"P{n:05d}"


def connect():
    missing = [k for k in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD") if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")
    return psycopg2.connect(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"), dbname=os.getenv("DB_NAME"),
                            user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
                            sslmode=os.getenv("DB_SSLMODE", "require"), connect_timeout=20)


def insert(cur, table, cols, rows):
    if rows:
        execute_values(cur, f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s", rows, page_size=2000)
    print(f"  {table:<20} {len(rows):>7} rows")


def build(n_patients):
    branches = [(f"B{i:02d}", f"MediAssist {c}", c, s, random.choice([150, 200, 300, 400]))
                for i, (c, s) in enumerate(CITIES[:8], 1)]
    depts = [(f"D{i:02d}", d) for i, d in enumerate(DEPARTMENTS, 1)]
    dept_id = {name: did for did, name in depts}
    insurance = [(f"INS{i:03d}", random.choice(INSURERS), random.choice(["Individual", "Family Floater", "Corporate"]),
                  random.choice([300000, 500000, 1000000])) for i in range(1, 41)]
    doctors = []
    for i in range(1, 61):
        dept = random.choice(DEPARTMENTS)
        doctors.append((f"DR{i:03d}", f"Dr. {random.choice(FIRST)} {random.choice(LAST)}", dept,
                        dept_id[dept], random.choice(branches)[0]))

    patients, admissions, appts, rx, labs, bills, pays = [], [], [], [], [], [], []
    adm_n = appt_n = rx_n = lab_n = bill_n = pay_n = 0
    for n in range(1, n_patients + 1):
        city, state = random.choice(CITIES)
        first, last = random.choice(FIRST), random.choice(LAST)
        age = random.randint(1, 90)
        insured = random.random() < 0.6
        ins_id = random.choice(insurance)[0] if insured else None
        branch = random.choice(branches)[0]
        reg = rand_date(date(2023, 1, 1), date(2026, 6, 30))
        patients.append((pid(n), first, last, random.choice(["Male", "Female"]),
                         iso(TODAY - timedelta(days=age * 365 + random.randint(0, 364))),
                         random.choice(BLOOD), f"9{random.randint(100000000, 999999999)}",
                         f"{first.lower()}.{last.lower()}{n}@example.com", city, state, iso(reg),
                         random.choices(["Active", "Inactive"], [92, 8])[0], ins_id, branch))

        for _ in range(random.choices([0, 1, 2, 3], [25, 45, 22, 8])[0]):  # admissions
            diag = random.choice(list(DIAGNOSES))
            dept, meds = DIAGNOSES[diag]
            doc = random.choice([d for d in doctors if d[2] == dept] or doctors)
            a_date = rand_date(max(reg, date(2024, 1, 1)), date(2026, 11, 20))
            stay = random.randint(1, 9)
            d_date = a_date + timedelta(days=stay)
            ongoing = d_date > TODAY
            adm_n += 1
            adm_id = f"ADM{adm_n:06d}"
            summary = (f"Admitted with {diag.lower()}. Treated per protocol and monitored for {stay} day(s). "
                       f"{'Currently admitted.' if ongoing else 'Discharged in stable condition with follow-up advised.'}")
            admissions.append((adm_id, pid(n), doc[0], dept_id[dept], branch, iso(a_date),
                               None if ongoing else iso(d_date), random.choice(ADM_TYPES), diag,
                               "Admitted" if ongoing else "Discharged", summary))
            for med in meds:  # prescriptions for this admission
                rx_n += 1
                rx.append((f"RX{rx_n:07d}", pid(n), doc[0], iso(a_date), med[0], med[1], med[2], med[3],
                           random.choice([5, 7, 10, 14, 30, 90]),
                           random.choice(["After food", "Before food", "At bedtime", "As directed by doctor"]),
                           random.choice(["Completed", "Active"])))
            total = round(random.uniform(8000, 250000), 2)  # billing
            covered = round(total * random.uniform(0.4, 0.9), 2) if ins_id else 0.0
            payable = round(total - covered, 2)
            status = random.choices(["Paid", "Partial", "Pending"], [65, 20, 15])[0]
            paid = payable if status == "Paid" else round(payable * random.uniform(0.2, 0.7), 2) if status == "Partial" else 0.0
            bill_n += 1
            bill_id = f"BILL{bill_n:06d}"
            mode = "Insurance" if ins_id and status != "Pending" and random.random() < 0.4 else random.choice(PAY_MODES)
            bills.append((bill_id, pid(n), adm_id, ins_id, iso(d_date if not ongoing else a_date),
                          total, covered, payable, paid, status, mode))
            if paid > 0:
                pay_n += 1
                pays.append((f"PAY{pay_n:06d}", bill_id, iso(min(d_date, TODAY)), paid, mode, "Success"))

        for _ in range(random.choices([1, 2, 4, 6], [20, 40, 30, 10])[0]):  # appointments
            doc = random.choice(doctors)
            ad = rand_date(date(2025, 1, 1), date(2027, 2, 28))
            st = "Scheduled" if ad > TODAY else random.choice(APPT_STATUS[:3] + APPT_STATUS[4:])
            appt_n += 1
            appts.append((f"APT{appt_n:07d}", pid(n), doc[0], branch, iso(ad),
                          f"{random.randint(9, 17):02d}:{random.choice(['00', '15', '30', '45'])}",
                          random.choice(APPT_TYPES), st,
                          None if st in ("Scheduled", "Cancelled", "No-show") else "Reviewed. Continue current treatment; follow up as advised."))

        for _ in range(random.choices([2, 4, 6, 8], [20, 40, 30, 10])[0]):  # lab results
            name, cat, unit, lo, hi = random.choice(LABS)
            roll = random.random()
            val = round(random.uniform(lo, hi), 1)
            if roll < 0.28:  # abnormal high
                val = round(hi * random.uniform(1.1, 2.0), 1) if hi else val
            elif roll < 0.36 and lo > 0:  # abnormal low
                val = round(lo * random.uniform(0.5, 0.9), 1)
            stat = "Normal" if lo <= val <= hi else ("High" if val > hi else "Low")
            if stat != "Normal" and (val > hi * 1.6 or (lo and val < lo * 0.6)):
                stat = "Critical"
            lab_n += 1
            labs.append((f"LAB{lab_n:07d}", pid(n), name, cat, iso(rand_date(date(2025, 1, 1), date(2026, 11, 25))),
                         str(val), unit, lo, hi, stat, random.choice(LAB_NAMES),
                         None if stat == "Normal" else "Abnormal value, clinical correlation advised."))

    return dict(branches=branches, depts=depts, insurance=insurance, doctors=doctors, patients=patients,
                admissions=admissions, appts=appts, rx=rx, labs=labs, bills=bills, pays=pays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=2000)
    n = ap.parse_args().patients
    data = build(n)
    schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    conn = connect()
    try:
        with conn, conn.cursor() as cur:
            print("Creating tables...")
            cur.execute(schema)
            print("Loading synthetic data...")
            insert(cur, "hospital_branches", ["branch_id", "branch_name", "city", "state", "bed_capacity"], data["branches"])
            insert(cur, "departments", ["department_id", "department_name"], data["depts"])
            insert(cur, "insurance_policies", ["insurance_id", "provider_name", "policy_type", "coverage_limit"], data["insurance"])
            insert(cur, "doctors", ["doctor_id", "doctor_name", "specialization", "department_id", "branch_id"], data["doctors"])
            insert(cur, "patients", ["patient_id", "first_name", "last_name", "gender", "date_of_birth", "blood_group", "phone",
                                     "email", "city", "state", "registration_date", "patient_status", "insurance_id", "branch_id"], data["patients"])
            insert(cur, "admissions", ["admission_id", "patient_id", "doctor_id", "department_id", "branch_id", "admission_date",
                                       "discharge_date", "admission_type", "diagnosis", "admission_status", "discharge_summary"], data["admissions"])
            insert(cur, "appointments", ["appointment_id", "patient_id", "doctor_id", "branch_id", "appointment_date", "appointment_time",
                                         "appointment_type", "appointment_status", "consultation_notes"], data["appts"])
            insert(cur, "prescriptions", ["prescription_id", "patient_id", "doctor_id", "prescription_date", "medicine_name", "dosage",
                                          "frequency", "route", "duration_days", "instructions", "prescription_status"], data["rx"])
            insert(cur, "lab_results", ["lab_result_id", "patient_id", "test_name", "test_category", "test_date", "result_value",
                                        "result_unit", "reference_min", "reference_max", "result_status", "lab_name", "report_notes"], data["labs"])
            insert(cur, "billing", ["bill_id", "patient_id", "admission_id", "insurance_id", "bill_date", "total_amount",
                                    "insurance_covered", "patient_payable", "amount_paid", "payment_status", "payment_mode"], data["bills"])
            insert(cur, "payments", ["payment_id", "bill_id", "payment_date", "amount", "payment_mode", "payment_status"], data["pays"])
    finally:
        conn.close()
    print("Done. Try asking the app: Show patient history for P00001")


if __name__ == "__main__":
    main()
