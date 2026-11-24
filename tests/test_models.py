"""Tests for the scikit-learn modeling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features.build import build_features, build_preprocessor


def test_logistic_pipeline_predicts_probabilities() -> None:
    rows = []
    labels = []
    for index in range(60):
        record = dict(config.DEFAULTS)
        record.update(
            {
                "age": "[60-70)" if index % 2 else "[30-40)",
                "time_in_hospital": 2 + index % 10,
                "num_medications": 5 + index % 20,
                "number_inpatient": index % 3,
                "diag_1": "250" if index % 2 else "428",
                "insulin": "Up" if index % 3 == 0 else "No",
            }
        )
        rows.append(record)
        labels.append(index % 2)

    features = build_features(pd.DataFrame(rows))
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    pipeline.fit(features, np.array(labels))

    probabilities = pipeline.predict_proba(features)[:, 1]
    assert probabilities.shape == (60,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert pipeline.predict(features).shape == (60,)
