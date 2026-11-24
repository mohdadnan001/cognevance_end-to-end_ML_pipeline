"""Tests for feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features.build import build_features, build_preprocessor, get_feature_names


def _record(**overrides) -> dict:
    record = dict(config.DEFAULTS)
    record.update(overrides)
    return record


def test_build_features_schema_and_engineering() -> None:
    frame = pd.DataFrame(
        [
            _record(
                age="[60-70)",
                diag_1="250.83",
                diag_2="V57",
                diag_3="?",
                insulin="Up",
                metformin="Steady",
                time_in_hospital=5,
                num_medications=20,
                number_inpatient=2,
                number_emergency=1,
                number_outpatient=0,
                num_lab_procedures=50,
                num_procedures=5,
            ),
            _record(age="[30-40)", diag_1="428", diag_2="401", diag_3="486"),
        ]
    )

    features = build_features(frame)

    assert list(features.columns) == config.FEATURE_COLUMNS
    assert "readmitted" not in features.columns
    assert "patient_nbr" not in features.columns
    assert features.loc[0, "age_numeric"] == 65
    assert features.loc[1, "age_numeric"] == 35
    assert features.loc[0, "diag_1_group"] == "Diabetes"
    assert features.loc[0, "diag_2_group"] == "Other"
    assert features.loc[0, "diag_3_group"] == "Unknown"
    assert features.loc[1, "diag_1_group"] == "Circulatory"
    assert features.loc[0, "n_active_meds"] == 2
    assert features.loc[0, "n_meds_changed"] == 1
    assert features.loc[0, "prior_visits_total"] == 3
    assert features.loc[0, "high_utilizer"] == 1
    assert features.loc[0, "polypharmacy"] == 1
    assert features.loc[0, "has_prior_inpatient"] == 1
    assert features.loc[0, "procedures_per_day"] == 11.0
    assert features.loc[0, "meds_per_day"] == 4.0


def test_build_features_handles_missing_columns() -> None:
    features = build_features(pd.DataFrame([{"age": "[50-60)", "diag_1": "250"}]))
    assert len(features) == 1
    assert features.loc[0, "age_numeric"] == 55
    assert features.notna().all().all()


def test_preprocessor_outputs_finite_matrix() -> None:
    frame = pd.DataFrame(
        [
            _record(age="[40-50)", diag_1="250", insulin="Steady"),
            _record(age="[70-80)", diag_1="428", insulin="No"),
        ]
    )
    features = build_features(frame)
    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(features)

    assert matrix.shape[0] == 2
    assert matrix.shape[1] == len(get_feature_names(preprocessor))
    assert np.isfinite(matrix).all()
