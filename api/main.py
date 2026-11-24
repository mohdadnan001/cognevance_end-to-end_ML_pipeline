"""FastAPI service exposing the readmission prediction model."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.models.predict import ReadmissionPredictor

LOGGER = logging.getLogger("readmission.api")
MAX_BATCH_ROWS = 10_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = ReadmissionPredictor()
        LOGGER.info("Loaded model: %s", app.state.predictor.model_name)
    except Exception as exc:
        app.state.predictor = None
        LOGGER.warning("Prediction model could not be loaded: %s", exc)
    yield


app = FastAPI(
    title="30-Day Hospital Readmission Prediction API",
    description=(
        "Predictive analytics service for diabetes patient encounters. "
        "Scores the probability of readmission within 30 days of discharge."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _predictor(request: Request) -> ReadmissionPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded. Run the training pipeline first.",
        )
    return predictor


@app.get("/", tags=["service"])
def root() -> dict:
    return {
        "service": "30-Day Hospital Readmission Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_name=getattr(predictor, "model_name", None),
    )


@app.get("/model/info", tags=["service"])
def model_info(request: Request) -> dict:
    return _predictor(request).info()


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: PatientFeatures, request: Request) -> PredictionResponse:
    predictor = _predictor(request)
    result = predictor.predict_records([payload.as_record()])[0]
    return PredictionResponse(**result)


@app.post("/predict/batch", tags=["prediction"])
async def predict_batch(
    request: Request, file: Annotated[UploadFile, File()]
) -> StreamingResponse:
    predictor = _predictor(request)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=400, detail="CSV contains no rows.")
    if len(frame) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size exceeds the {MAX_BATCH_ROWS:,} row limit.",
        )
    try:
        scored = predictor.predict_dataframe(frame)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Scoring failed: {exc}") from exc

    buffer = io.StringIO()
    scored.to_csv(buffer, index=False)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="readmission_predictions.csv"'},
    )
