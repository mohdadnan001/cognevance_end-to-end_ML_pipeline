"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


class DummyPredictor:
    model_name = "dummy_model"
    threshold = 0.5

    def predict_records(self, records):
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        return [
            {
                "probability": 0.2,
                "prediction": 0,
                "risk_level": "low",
                "threshold": self.threshold,
            }
            for _ in range(len(frame))
        ]

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["probability"] = 0.2
        output["prediction"] = 0
        output["risk_level"] = "low"
        return output

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": "sklearn",
            "threshold": self.threshold,
            "metrics": {"test": {"pr_auc": 0.0}},
        }


@pytest.fixture()
def dummy_predictor() -> DummyPredictor:
    return DummyPredictor()
