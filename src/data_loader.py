"""Load case vignettes from the source xlsx."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openpyxl


SPECIALTY_MAP = {
    "CARDIO": "cardiology",
    "PULMO": "pulmonology",
    "ORTHO": "orthopedics",
    "NEURO": "neurology",
    "PED": "pediatrics",
}


@dataclass(frozen=True)
class Case:
    case_id: str
    specialty: str
    patient_profile: str
    chief_complaint: str
    medical_history: str
    vital_signs: str
    physical_examination: str
    diagnostics: str
    correct_diagnosis: str
    dd_a: str
    dd_b: str
    dd_c: str
    dd_d: str


def _specialty_from_case_id(case_id: str) -> str:
    prefix = case_id.split("_", 1)[0]
    if prefix not in SPECIALTY_MAP:
        raise ValueError(
            f"Unknown specialty prefix {prefix!r} in case_id {case_id!r}. "
            f"Known prefixes: {sorted(SPECIALTY_MAP)}"
        )
    return SPECIALTY_MAP[prefix]


def load_cases(xlsx_path: str | Path, sheet: str = "Tabelle1") -> list[Case]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet]

    expected_headers = [
        "Heading", "Case ID", "Patient Profile", "Chief Complaint",
        "Medical History", "Vital Signs", "Physical Examination", "Diagnostics",
        "Correct_Diagnosis", "Distractor (DD_a)", "DD_b", "DD_c", "DD_d",
    ]
    actual_headers = [c.value for c in ws[1][:len(expected_headers)]]
    if actual_headers != expected_headers:
        raise ValueError(
            f"Unexpected header row.\n  expected: {expected_headers}\n  actual:   {actual_headers}"
        )

    cases: list[Case] = []
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        # Drop trailing template columns (Neutral / Sycophancy_Mild / Sycophancy_Severe)
        values = row[:13]
        if all(v is None for v in values):
            continue  # tolerate trailing blank rows

        (heading, case_id, patient_profile, chief_complaint, medical_history,
         vital_signs, physical_examination, diagnostics, correct_diagnosis,
         dd_a, dd_b, dd_c, dd_d) = values

        required = {
            "Case ID": case_id,
            "Patient Profile": patient_profile,
            "Chief Complaint": chief_complaint,
            "Medical History": medical_history,
            "Vital Signs": vital_signs,
            "Physical Examination": physical_examination,
            "Diagnostics": diagnostics,
            "Correct_Diagnosis": correct_diagnosis,
            "Distractor (DD_a)": dd_a,
            "DD_b": dd_b,
            "DD_c": dd_c,
            "DD_d": dd_d,
        }
        missing = [k for k, v in required.items() if v is None or str(v).strip() == ""]
        if missing:
            raise ValueError(f"Row {row_idx} ({case_id!r}) is missing required fields: {missing}")

        case_id_str = str(case_id).strip()
        cases.append(Case(
            case_id=case_id_str,
            specialty=_specialty_from_case_id(case_id_str),
            patient_profile=str(patient_profile).strip(),
            chief_complaint=str(chief_complaint).strip(),
            medical_history=str(medical_history).strip(),
            vital_signs=str(vital_signs).strip(),
            physical_examination=str(physical_examination).strip(),
            diagnostics=str(diagnostics).strip(),
            correct_diagnosis=str(correct_diagnosis).strip(),
            dd_a=str(dd_a).strip(),
            dd_b=str(dd_b).strip(),
            dd_c=str(dd_c).strip(),
            dd_d=str(dd_d).strip(),
        ))

    return cases


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "case_vignettes.xlsx"
    cases = load_cases(path)
    print(f"Loaded {len(cases)} cases.")
    print(f"First case: {cases[0].case_id} ({cases[0].specialty}) — correct: {cases[0].correct_diagnosis!r}")
    from collections import Counter
    print("Specialty counts:", Counter(c.specialty for c in cases))
