"""Unified inference wrapper for scikit-learn and Keras artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features.build import build_features


class ReadmissionPredictor:
    """Load trained artifacts and score raw patient records."""

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir is not None else config.MODELS_DIR
        metadata_path = self.models_dir / config.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"{metadata_path} is missing. Run `python scripts/run_pipeline.py` first."
            )

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model_name: str = self.metadata["model_name"]
        self.model_type: str = self.metadata["model_type"]
        self.threshold: float = float(self.metadata["threshold"])
        self.preprocessor = joblib.load(self.models_dir / config.PREPROCESSOR_FILE)

        if self.model_type == "keras":
            import os

            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from tensorflow import keras

            self.model = keras.models.load_model(self.models_dir / config.KERAS_MODEL_FILE)
        else:
            self.model = joblib.load(self.models_dir / config.BEST_MODEL_FILE)

    def _probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        features = build_features(frame)
        if self.model_type == "keras":
            transformed = self.preprocessor.transform(features)
            probabilities = self.model.predict(transformed, verbose=0).ravel()
        else:
            probabilities = self.model.predict_proba(features)[:, 1]
        return np.asarray(probabilities, dtype=float)

    def _risk_level(self, probability: float) -> str:
        if probability < self.threshold:
            return "low"
        if probability < min(2.0 * self.threshold, 0.95):
            return "medium"
        return "high"

    def _format_prediction(self, probability: float) -> dict:
        return {
            "probability": round(float(probability), 6),
            "prediction": int(probability >= self.threshold),
            "risk_level": self._risk_level(float(probability)),
            "threshold": round(self.threshold, 6),
        }

    def predict_records(self, records: list[dict] | pd.DataFrame) -> list[dict]:
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        probabilities = self._probabilities(frame)
        return [self._format_prediction(probability) for probability in probabilities]

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self._probabilities(frame)
        output = frame.copy()
        output["probability"] = probabilities
        output["prediction"] = (probabilities >= self.threshold).astype(int)
        output["risk_level"] = [self._risk_level(float(value)) for value in probabilities]
        return output

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "threshold": round(self.threshold, 4),
            "trained_at_utc": self.metadata.get("trained_at_utc"),
            "dataset": self.metadata.get("dataset"),
            "metrics": self.metadata.get("metrics"),
            "n_features_transformed": self.metadata.get("features", {}).get("n_transformed"),
            "versions": self.metadata.get("versions"),
        }
