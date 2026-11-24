"""Feature engineering and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

MISSING_TOKENS = {"?", "", "nan", "NaN", "<NA>", "null", "NULL"}


def _as_category(series: pd.Series) -> pd.Series:
    """Normalize a categorical column to clean strings with ``Unknown`` fills."""
    values = series.astype("object").where(series.notna(), "Unknown")
    values = values.astype(str).str.strip()
    return values.mask(values.isin(MISSING_TOKENS), "Unknown")


def _map_age(series: pd.Series) -> pd.Series:
    cleaned = _as_category(series)
    return cleaned.map(config.AGE_MIDPOINTS).astype(float)


def _map_diagnosis(code: object) -> str:
    """Map an ICD-9 code to one of nine clinical groups."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Unknown"
    text = str(code).strip()
    if text in MISSING_TOKENS:
        return "Unknown"
    if text[0] in {"V", "E"}:
        return "Other"
    try:
        value = float(text)
    except ValueError:
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value <= 459 or 785 <= value < 786:
        return "Circulatory"
    if 460 <= value <= 519 or 786 <= value < 787:
        return "Respiratory"
    if 520 <= value <= 579 or 787 <= value < 788:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or 788 <= value < 789:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.where(denominator > 0)
    return numerator / safe_denominator


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Transform raw patient records into the model feature schema.

    The function is defensive: missing columns are filled with clinical
    defaults so that single-record API requests remain valid.
    """
    data = frame.copy()

    for column, default in config.DEFAULTS.items():
        if column not in data.columns:
            data[column] = default

    for column in config.NUMERIC_RAW:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in [
        "race",
        "gender",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "diabetesMed",
        "change",
        *config.MEDICATION_COLUMNS,
    ]:
        data[column] = _as_category(data[column])

    data["age_numeric"] = _map_age(data["age"])
    data["diag_1_group"] = data["diag_1"].map(_map_diagnosis)
    data["diag_2_group"] = data["diag_2"].map(_map_diagnosis)
    data["diag_3_group"] = data["diag_3"].map(_map_diagnosis)

    medication_values = data[config.MEDICATION_COLUMNS]
    data["n_active_meds"] = medication_values.isin(["Steady", "Up", "Down"]).sum(axis=1)
    data["n_meds_changed"] = medication_values.isin(["Up", "Down"]).sum(axis=1)
    data["n_meds_up"] = (medication_values == "Up").sum(axis=1)
    data["n_meds_down"] = (medication_values == "Down").sum(axis=1)

    outpatient = data["number_outpatient"].fillna(0)
    emergency = data["number_emergency"].fillna(0)
    inpatient = data["number_inpatient"].fillna(0)
    prior_visits = outpatient + emergency + inpatient

    data["prior_visits_total"] = prior_visits
    data["inpatient_ratio"] = _safe_ratio(inpatient, prior_visits + 1)
    data["emergency_ratio"] = _safe_ratio(emergency, prior_visits + 1)
    data["outpatient_ratio"] = _safe_ratio(outpatient, prior_visits + 1)

    days = data["time_in_hospital"].clip(lower=1)
    data["procedures_per_day"] = (
        data["num_lab_procedures"].fillna(0) + data["num_procedures"].fillna(0)
    ) / days
    data["meds_per_day"] = data["num_medications"].fillna(0) / days

    data["has_prior_inpatient"] = (inpatient > 0).astype(int)
    data["high_utilizer"] = (prior_visits >= 3).astype(int)
    data["polypharmacy"] = (data["num_medications"].fillna(0) >= 15).astype(int)

    features = data[config.FEATURE_COLUMNS].copy()
    for column in config.MODEL_NUMERIC:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    for column in config.MODEL_CATEGORICAL:
        features[column] = features[column].astype(str)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def build_preprocessor() -> ColumnTransformer:
    """Create the numeric/categorical preprocessing transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.MODEL_NUMERIC),
            ("cat", categorical_pipeline, config.MODEL_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return readable output feature names from a fitted preprocessor."""
    names = preprocessor.get_feature_names_out()
    return [str(name).split("__", 1)[-1] for name in names]
