"""Central configuration for the readmission prediction pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
DATASET_DOI = "10.24432/C5230J"
DATASET_NAME = "Diabetes 130-US Hospitals for Years 1999-2008"
DATASET_LICENSE = "CC BY 4.0"

RANDOM_STATE = 42
N_SPLITS = 7
TARGET_COLUMN = "readmitted"
TARGET_NAME = "readmitted_30d"
GROUP_COLUMN = "patient_nbr"

PREPROCESSOR_FILE = "preprocessor.joblib"
BEST_MODEL_FILE = "best_model.joblib"
KERAS_MODEL_FILE = "keras_model.keras"
METADATA_FILE = "metadata.json"

ID_COLUMNS = ["encounter_id", "patient_nbr"]
HIGH_MISSING_COLUMNS = ["weight", "payer_code"]
DROP_COLUMNS = ["encounter_id", "patient_nbr", "weight", "payer_code", TARGET_COLUMN]

NUMERIC_RAW = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

ENGINEERED_NUMERIC = [
    "age_numeric",
    "prior_visits_total",
    "inpatient_ratio",
    "emergency_ratio",
    "outpatient_ratio",
    "procedures_per_day",
    "meds_per_day",
    "n_active_meds",
    "n_meds_changed",
    "n_meds_up",
    "n_meds_down",
    "has_prior_inpatient",
    "high_utilizer",
    "polypharmacy",
]

MODEL_NUMERIC = NUMERIC_RAW + ENGINEERED_NUMERIC

MODEL_CATEGORICAL = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "diabetesMed",
    "change",
]

FEATURE_COLUMNS = MODEL_NUMERIC + MODEL_CATEGORICAL

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

DIAG_GROUPS = [
    "Diabetes",
    "Circulatory",
    "Respiratory",
    "Digestive",
    "Injury",
    "Musculoskeletal",
    "Genitourinary",
    "Neoplasms",
    "Other",
    "Unknown",
]

CATEGORICAL_DEFAULTS = {
    "race": "Unknown",
    "gender": "Female",
    "age": "[60-70)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "medical_specialty": "Unknown",
    "diag_1": "250",
    "diag_2": "250",
    "diag_3": "250",
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "insulin": "No",
    "diabetesMed": "Yes",
    "change": "No",
}

NUMERIC_DEFAULTS = {
    "time_in_hospital": 4,
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 16,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 9,
}

DEFAULTS = {
    **CATEGORICAL_DEFAULTS,
    **NUMERIC_DEFAULTS,
    **{column: "No" for column in MEDICATION_COLUMNS},
}

TARGET_LABELS = {0: "No 30-day readmission", 1: "30-day readmission"}


def ensure_directories() -> None:
    """Create all project directories used by the pipeline."""
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
