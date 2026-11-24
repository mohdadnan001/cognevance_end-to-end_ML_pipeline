"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgeBand = Literal[
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
MedicationStatus = Literal["No", "Steady", "Up", "Down"]


class PatientFeatures(BaseModel):
    """Raw clinical features required to score a patient encounter."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[70-80)",
                "time_in_hospital": 6,
                "num_medications": 22,
                "number_inpatient": 2,
                "number_emergency": 1,
                "number_diagnoses": 9,
                "diag_1": "428",
                "diag_2": "250",
                "diag_3": "401",
                "insulin": "Up",
                "A1Cresult": ">7",
                "change": "Ch",
                "diabetesMed": "Yes",
            }
        },
    )

    race: Literal["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"] = Field(
        "Caucasian", description="Patient race"
    )
    gender: Literal["Female", "Male", "Unknown"] = Field("Female", description="Patient gender")
    age: AgeBand = Field("[60-70)", description="10-year age band")

    admission_type_id: int = Field(1, description="Admission type code (1=emergency, 2=urgent, ...)")
    discharge_disposition_id: int = Field(1, description="Discharge disposition code")
    admission_source_id: int = Field(7, description="Admission source code (7=emergency room)")
    time_in_hospital: int = Field(4, ge=1, le=14, description="Days between admission and discharge")
    medical_specialty: str = Field("Unknown", description="Admitting physician specialty")

    num_lab_procedures: int = Field(44, ge=0, description="Number of lab procedures")
    num_procedures: int = Field(1, ge=0, description="Number of non-lab procedures")
    num_medications: int = Field(16, ge=0, description="Number of distinct medications")
    number_outpatient: int = Field(0, ge=0, description="Outpatient visits in the prior year")
    number_emergency: int = Field(0, ge=0, description="Emergency visits in the prior year")
    number_inpatient: int = Field(0, ge=0, description="Inpatient visits in the prior year")
    number_diagnoses: int = Field(9, ge=1, le=16, description="Number of diagnoses on record")

    diag_1: str = Field("250", description="Primary ICD-9 diagnosis code")
    diag_2: str = Field("250", description="Secondary ICD-9 diagnosis code")
    diag_3: str = Field("250", description="Additional ICD-9 diagnosis code")

    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = Field(
        "None", description="Maximum glucose serum result"
    )
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = Field("None", description="HbA1c result")

    metformin: MedicationStatus = "No"
    repaglinide: MedicationStatus = "No"
    nateglinide: MedicationStatus = "No"
    chlorpropamide: MedicationStatus = "No"
    glimepiride: MedicationStatus = "No"
    acetohexamide: MedicationStatus = "No"
    glipizide: MedicationStatus = "No"
    glyburide: MedicationStatus = "No"
    tolbutamide: MedicationStatus = "No"
    pioglitazone: MedicationStatus = "No"
    rosiglitazone: MedicationStatus = "No"
    acarbose: MedicationStatus = "No"
    miglitol: MedicationStatus = "No"
    troglitazone: MedicationStatus = "No"
    tolazamide: MedicationStatus = "No"
    examide: MedicationStatus = "No"
    citoglipton: MedicationStatus = "No"
    insulin: MedicationStatus = "No"
    glyburide_metformin: MedicationStatus = Field("No", alias="glyburide-metformin")
    glipizide_metformin: MedicationStatus = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: MedicationStatus = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedicationStatus = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedicationStatus = Field("No", alias="metformin-pioglitazone")

    change: Literal["No", "Ch"] = Field("No", description="Diabetes medication change")
    diabetesMed: Literal["No", "Yes"] = Field("Yes", description="Diabetes medication prescribed")

    def as_record(self) -> dict:
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Predicted probability of 30-day readmission")
    prediction: int = Field(..., description="Binary prediction at the tuned threshold")
    risk_level: str = Field(..., description="Low, medium or high risk")
    threshold: float = Field(..., description="Operating threshold selected on validation data")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
