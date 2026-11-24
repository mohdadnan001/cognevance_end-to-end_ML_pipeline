"""Smoke test for a running prediction API."""

from __future__ import annotations

import io
import os
import sys

import pandas as pd
import requests

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

PAYLOAD = {
    "age": "[70-80)",
    "gender": "Female",
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


def main() -> int:
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    health.raise_for_status()
    print("health:", health.json())
    assert health.json()["model_loaded"], "Model is not loaded"

    info = requests.get(f"{BASE_URL}/model/info", timeout=10)
    info.raise_for_status()
    print("model:", info.json()["model_name"], "threshold:", info.json()["threshold"])

    prediction = requests.post(f"{BASE_URL}/predict", json=PAYLOAD, timeout=10)
    prediction.raise_for_status()
    print("predict:", prediction.json())

    frame = pd.DataFrame(
        [
            PAYLOAD,
            {**PAYLOAD, "age": "[60-70)", "time_in_hospital": 4, "insulin": "No"},
            {**PAYLOAD, "age": "[80-90)", "time_in_hospital": 9, "number_inpatient": 0},
        ]
    )
    buffer = frame.to_csv(index=False).encode("utf-8")
    batch = requests.post(
        f"{BASE_URL}/predict/batch",
        files={"file": ("batch.csv", buffer, "text/csv")},
        timeout=30,
    )
    batch.raise_for_status()
    scored = pd.read_csv(io.StringIO(batch.text))
    print("batch rows:", len(scored), "columns:", list(scored.columns)[-4:])
    assert {"probability", "prediction", "risk_level"}.issubset(scored.columns)

    print("API smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
