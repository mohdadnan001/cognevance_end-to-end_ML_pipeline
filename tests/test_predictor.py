"""Smoke tests against the trained artifacts (skipped before training)."""

from __future__ import annotations

import pytest

from src import config


@pytest.mark.skipif(
    not (config.MODELS_DIR / config.METADATA_FILE).exists(),
    reason="Trained artifacts are not available yet.",
)
def test_predictor_scores_default_record() -> None:
    from src.models.predict import ReadmissionPredictor

    predictor = ReadmissionPredictor()
    result = predictor.predict_records([dict(config.DEFAULTS)])[0]

    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["risk_level"] in {"low", "medium", "high"}
    assert result["threshold"] == pytest.approx(predictor.threshold, abs=1e-6)
