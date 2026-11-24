"""API contract tests using a dummy predictor."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def _client(dummy_predictor) -> TestClient:
    app.state.predictor = dummy_predictor
    return TestClient(app)


def test_health_reports_model(dummy_predictor) -> None:
    client = _client(dummy_predictor)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_name"] == "dummy_model"


def test_predict_with_defaults(dummy_predictor) -> None:
    client = _client(dummy_predictor)
    response = client.post("/predict", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["probability"] == 0.2
    assert payload["prediction"] == 0
    assert payload["risk_level"] == "low"


def test_predict_accepts_hyphenated_medication_alias(dummy_predictor) -> None:
    client = _client(dummy_predictor)
    response = client.post(
        "/predict",
        json={"age": "[70-80)", "insulin": "Up", "glyburide-metformin": "Steady"},
    )
    assert response.status_code == 200


def test_predict_rejects_invalid_age(dummy_predictor) -> None:
    client = _client(dummy_predictor)
    response = client.post("/predict", json={"age": "not-an-age"})
    assert response.status_code == 422


def test_model_info(dummy_predictor) -> None:
    client = _client(dummy_predictor)
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["model_name"] == "dummy_model"


def test_batch_prediction_returns_csv(dummy_predictor) -> None:
    client = _client(dummy_predictor)
    csv = "age,gender,time_in_hospital\n[60-70),Female,4\n[70-80),Male,8\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("patients.csv", csv.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    body = response.text
    assert "probability" in body
    assert body.count("\n") >= 3


def test_missing_model_returns_503() -> None:
    app.state.predictor = None
    client = TestClient(app)
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/predict", json={}).status_code == 503
