-- MediAssist AI: hospital database schema (matches what app/mcp/tools.py queries).
-- Dates are stored as TEXT in YYYY-MM-DD form, as the MCP tools expect.

DROP TABLE IF EXISTS payments, billing, lab_results, prescriptions, appointments,
                     admissions, patients, doctors, insurance_policies,
                     departments, hospital_branches CASCADE;

CREATE TABLE hospital_branches (
    branch_id     TEXT PRIMARY KEY,
    branch_name   TEXT NOT NULL,
    city          TEXT NOT NULL,
    state         TEXT NOT NULL,
    bed_capacity  INTEGER
);

CREATE TABLE departments (
    department_id    TEXT PRIMARY KEY,
    department_name  TEXT NOT NULL
);

CREATE TABLE insurance_policies (
    insurance_id     TEXT PRIMARY KEY,
    provider_name    TEXT NOT NULL,
    policy_type      TEXT,
    coverage_limit   DOUBLE PRECISION
);

CREATE TABLE doctors (
    doctor_id       TEXT PRIMARY KEY,
    doctor_name     TEXT NOT NULL,
    specialization  TEXT NOT NULL,
    department_id   TEXT REFERENCES departments(department_id),
    branch_id       TEXT REFERENCES hospital_branches(branch_id)
);

CREATE TABLE patients (
    patient_id         TEXT PRIMARY KEY,          -- P00001
    first_name         TEXT NOT NULL,
    last_name          TEXT NOT NULL,
    gender             TEXT,
    date_of_birth      TEXT,
    blood_group        TEXT,
    phone              TEXT,
    email              TEXT,
    city               TEXT,
    state              TEXT,
    registration_date  TEXT,
    patient_status     TEXT,
    insurance_id       TEXT REFERENCES insurance_policies(insurance_id),
    branch_id          TEXT REFERENCES hospital_branches(branch_id)
);

CREATE TABLE admissions (
    admission_id       TEXT PRIMARY KEY,
    patient_id         TEXT NOT NULL REFERENCES patients(patient_id),
    doctor_id          TEXT REFERENCES doctors(doctor_id),
    department_id      TEXT REFERENCES departments(department_id),
    branch_id          TEXT REFERENCES hospital_branches(branch_id),
    admission_date     TEXT,
    discharge_date     TEXT,
    admission_type     TEXT,
    diagnosis          TEXT,
    admission_status   TEXT,
    discharge_summary  TEXT
);

CREATE TABLE appointments (
    appointment_id      TEXT PRIMARY KEY,
    patient_id          TEXT NOT NULL REFERENCES patients(patient_id),
    doctor_id           TEXT REFERENCES doctors(doctor_id),
    branch_id           TEXT REFERENCES hospital_branches(branch_id),
    appointment_date    TEXT,
    appointment_time    TEXT,
    appointment_type    TEXT,
    appointment_status  TEXT,
    consultation_notes  TEXT
);

CREATE TABLE prescriptions (
    prescription_id      TEXT PRIMARY KEY,
    patient_id           TEXT NOT NULL REFERENCES patients(patient_id),
    doctor_id            TEXT REFERENCES doctors(doctor_id),
    prescription_date    TEXT,
    medicine_name        TEXT,
    dosage               TEXT,
    frequency            TEXT,
    route                TEXT,
    duration_days        INTEGER,
    instructions         TEXT,
    prescription_status  TEXT
);

CREATE TABLE lab_results (
    lab_result_id   TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL REFERENCES patients(patient_id),
    test_name       TEXT,
    test_category   TEXT,
    test_date       TEXT,
    result_value    TEXT,
    result_unit     TEXT,
    reference_min   DOUBLE PRECISION,
    reference_max   DOUBLE PRECISION,
    result_status   TEXT,
    lab_name        TEXT,
    report_notes    TEXT
);

CREATE TABLE billing (
    bill_id            TEXT PRIMARY KEY,
    patient_id         TEXT NOT NULL REFERENCES patients(patient_id),
    admission_id       TEXT REFERENCES admissions(admission_id),
    insurance_id       TEXT REFERENCES insurance_policies(insurance_id),
    bill_date          TEXT,
    total_amount       DOUBLE PRECISION,
    insurance_covered  DOUBLE PRECISION,
    patient_payable    DOUBLE PRECISION,
    amount_paid        DOUBLE PRECISION,
    payment_status     TEXT,
    payment_mode       TEXT
);

CREATE TABLE payments (
    payment_id      TEXT PRIMARY KEY,
    bill_id         TEXT NOT NULL REFERENCES billing(bill_id),
    payment_date    TEXT,
    amount          DOUBLE PRECISION,
    payment_mode    TEXT,
    payment_status  TEXT
);

CREATE INDEX idx_adm_patient   ON admissions(patient_id);
CREATE INDEX idx_appt_patient  ON appointments(patient_id);
CREATE INDEX idx_rx_patient    ON prescriptions(patient_id);
CREATE INDEX idx_lab_patient   ON lab_results(patient_id);
CREATE INDEX idx_bill_patient  ON billing(patient_id);
CREATE INDEX idx_pay_bill      ON payments(bill_id);
