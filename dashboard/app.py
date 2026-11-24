"""Streamlit dashboard for readmission analytics and live scoring."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

st.set_page_config(
    page_title="Readmission Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0f766e", "#f59e0b", "#b45309", "#134e4a", "#64748b"]

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.9rem; }
    .risk-badge {
        display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
        font-weight: 600; color: white; letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    path = config.METRICS_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame | None:
    from src.data.download import load_dataset

    try:
        return load_dataset()
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading local model...")
def load_local_predictor():
    from src.models.predict import ReadmissionPredictor

    return ReadmissionPredictor()


@st.cache_data(ttl=10, show_spinner=False)
def api_online() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=1.5)
        return response.ok and bool(response.json().get("model_loaded"))
    except requests.RequestException:
        return False


def predict_record(record: dict) -> tuple[dict, str]:
    if api_online():
        response = requests.post(f"{API_URL}/predict", json=record, timeout=15)
        response.raise_for_status()
        return response.json(), "FastAPI"
    return load_local_predictor().predict_records([record])[0], "local artifact"


def predict_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if api_online():
        buffer = frame.to_csv(index=False).encode("utf-8")
        response = requests.post(
            f"{API_URL}/predict/batch",
            files={"file": ("batch.csv", buffer, "text/csv")},
            timeout=180,
        )
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text)), "FastAPI"
    return load_local_predictor().predict_dataframe(frame), "local artifact"


def risk_color(level: str) -> str:
    return {"low": "#0f766e", "medium": "#d97706", "high": "#b91c1c"}.get(level, "#334155")


def render_overview(metadata: dict, eda: dict, comparison: pd.DataFrame) -> None:
    st.title("30-Day Readmission Analytics")
    st.caption(
        "End-to-end ML pipeline for diabetes patient encounters — "
        "UCI Diabetes 130-US Hospitals (1999–2008)."
    )

    best_metrics = metadata.get("metrics", {}).get("test", {})
    columns = st.columns(5)
    columns[0].metric("Encounters", f"{eda.get('rows', 0):,}")
    columns[1].metric("Patients", f"{eda.get('unique_patients', 0):,}")
    columns[2].metric("30-day readmission", f"{100 * eda.get('target_rate', 0):.1f}%")
    columns[3].metric("Best model", metadata.get("model_name", "n/a"))
    columns[4].metric("Test PR-AUC", f"{best_metrics.get('pr_auc', 0):.3f}")

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Deployed model")
        st.markdown(
            f"""
            - **Algorithm:** `{metadata.get('model_name', 'n/a')}` ({metadata.get('model_type', 'n/a')})
            - **Decision threshold:** `{metadata.get('threshold', 0.5):.2f}` (tuned for F1 on validation)
            - **Test recall:** `{best_metrics.get('recall', 0):.3f}`
            - **Test ROC-AUC:** `{best_metrics.get('roc_auc', 0):.3f}`
            - **Engineered features:** `{metadata.get('features', {}).get('n_transformed', 0)}`
            - **Trained:** `{metadata.get('trained_at_utc', 'n/a')}`
            """
        )
    with right:
        st.subheader("Pipeline stages")
        st.markdown(
            """
            1. Data acquisition & provenance
            2. Cleaning, ICD-9 grouping, utilization features
            3. Patient-aware stratified split (no leakage)
            4. Multi-model training & threshold tuning
            5. Evaluation, calibration & permutation importance
            6. FastAPI deployment + Streamlit analytics
            """
        )

    if not comparison.empty:
        st.subheader("Model leaderboard")
        display_columns = [
            column
            for column in ["model", "validation_pr_auc", "test_pr_auc", "test_roc_auc", "test_f1", "test_recall"]
            if column in comparison.columns
        ]
        st.dataframe(
            comparison[display_columns].rename(
                columns={
                    "model": "Model",
                    "validation_pr_auc": "Val PR-AUC",
                    "test_pr_auc": "Test PR-AUC",
                    "test_roc_auc": "Test ROC-AUC",
                    "test_f1": "Test F1",
                    "test_recall": "Test recall",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_data_insights(eda: dict) -> None:
    st.title("Data Insights")
    frame = load_raw_data()
    if frame is None:
        st.info("Raw dataset is not available. Run `python scripts/run_pipeline.py` first.")
        return

    target = (frame[config.TARGET_COLUMN].astype(str).str.strip() == "<30").astype(int)
    frame = frame.assign(target=target)

    filters = st.columns(3)
    ages = filters[0].multiselect("Age band", eda.get("age_order", []), default=[])
    genders = filters[1].multiselect("Gender", sorted(frame["gender"].unique()), default=[])
    insulin = filters[2].multiselect("Insulin", sorted(frame["insulin"].unique()), default=[])

    if ages:
        frame = frame[frame["age"].isin(ages)]
    if genders:
        frame = frame[frame["gender"].isin(genders)]
    if insulin:
        frame = frame[frame["insulin"].isin(insulin)]

    kpis = st.columns(4)
    kpis[0].metric("Filtered encounters", f"{len(frame):,}")
    kpis[1].metric("Readmission rate", f"{100 * frame['target'].mean():.2f}%")
    kpis[2].metric("Mean length of stay", f"{frame['time_in_hospital'].mean():.1f} days")
    kpis[3].metric("Mean medications", f"{frame['num_medications'].mean():.1f}")

    st.divider()
    left, right = st.columns(2)

    age_order = [band for band in eda.get("age_order", []) if band in set(frame["age"])]
    age_rate = frame.groupby("age", observed=True)["target"].mean().reindex(age_order).dropna()
    left.plotly_chart(
        px.bar(
            x=age_rate.index,
            y=100 * age_rate.values,
            labels={"x": "Age band", "y": "30-day readmission rate (%)"},
            title="Readmission rate by age",
        ),
        use_container_width=True,
    )

    time_rate = frame.groupby("time_in_hospital", observed=True)["target"].mean()
    right.plotly_chart(
        px.line(
            x=time_rate.index,
            y=100 * time_rate.values,
            markers=True,
            labels={"x": "Days in hospital", "y": "30-day readmission rate (%)"},
            title="Readmission rate by length of stay",
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)
    inpatient = frame["number_inpatient"].clip(upper=3)
    inpatient_rate = frame.assign(bucket=inpatient).groupby("bucket", observed=True)["target"].mean()
    left.plotly_chart(
        px.bar(
            x=[str(value) for value in inpatient_rate.index],
            y=100 * inpatient_rate.values,
            labels={"x": "Prior inpatient visits", "y": "30-day readmission rate (%)"},
            title="Readmission rate by prior inpatient utilization",
        ),
        use_container_width=True,
    )

    insulin_rate = frame.groupby("insulin", observed=True)["target"].mean().sort_values(ascending=False)
    right.plotly_chart(
        px.bar(
            x=insulin_rate.index,
            y=100 * insulin_rate.values,
            labels={"x": "Insulin status", "y": "30-day readmission rate (%)"},
            title="Readmission rate by insulin regimen",
        ),
        use_container_width=True,
    )

    st.subheader("Data quality")
    missing = frame.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(10)
    if missing.empty:
        st.success("No missing values in the filtered view.")
    else:
        st.plotly_chart(
            px.bar(
                x=100 * missing.values,
                y=missing.index,
                orientation="h",
                labels={"x": "Missing values (%)", "y": ""},
                title="Top missing columns",
            ),
            use_container_width=True,
        )

    st.subheader("Sample records")
    st.dataframe(
        frame.drop(columns=["target"]).head(25),
        use_container_width=True,
        hide_index=True,
    )


def render_model_performance(metadata: dict, comparison: pd.DataFrame) -> None:
    st.title("Model Performance")
    if comparison.empty:
        st.info("Metrics are not available. Train the models first.")
        return

    metric_columns = [column for column in ["test_pr_auc", "test_roc_auc", "test_f1", "test_recall"] if column in comparison]
    melted = comparison.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )
    st.plotly_chart(
        px.bar(
            melted,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            title="Test-set comparison",
            labels={"model": "", "score": "Score", "metric": "Metric"},
        ),
        use_container_width=True,
    )

    figures = [
        ("roc_curves.png", "ROC curves"),
        ("pr_curves.png", "Precision-recall curves"),
        ("confusion_matrix_best.png", f"Confusion matrix — {metadata.get('model_name', 'best model')}"),
        ("calibration_best.png", "Calibration"),
        ("feature_importance_best.png", "Permutation importance"),
        ("model_comparison.png", "Summary comparison"),
        ("keras_training_curves.png", "Keras training history"),
    ]
    available = [(name, caption) for name, caption in figures if (config.FIGURES_DIR / name).exists()]
    for index in range(0, len(available), 2):
        columns = st.columns(2)
        for column, (name, caption) in zip(
            columns, available[index : index + 2], strict=False
        ):
            column.image(str(config.FIGURES_DIR / name), caption=caption, use_container_width=True)

    st.subheader("All metrics")
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def render_predict(metadata: dict) -> None:
    st.title("Patient Risk Scoring")
    st.caption(
        "Enter encounter details to estimate the probability of readmission within 30 days. "
        f"Scoring source: {'FastAPI' if api_online() else 'local artifact'}."
    )

    with st.form("patient_form"):
        left, middle, right = st.columns(3)

        with left:
            st.markdown("**Demographics & admission**")
            age = st.selectbox("Age band", list(config.AGE_MIDPOINTS.keys()), index=7)
            gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])
            race = st.selectbox(
                "Race",
                ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"],
            )
            admission_type = st.number_input("Admission type ID", 1, 9, 1)
            discharge = st.number_input("Discharge disposition ID", 1, 30, 1)
            source = st.number_input("Admission source ID", 1, 26, 7)
            specialty = st.text_input("Medical specialty", "Unknown")

        with middle:
            st.markdown("**Utilization**")
            time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
            lab_procedures = st.number_input("Lab procedures", 0, 200, 44)
            procedures = st.number_input("Other procedures", 0, 50, 1)
            medications = st.number_input("Medications", 0, 100, 16)
            outpatient = st.number_input("Prior outpatient visits", 0, 50, 0)
            emergency = st.number_input("Prior emergency visits", 0, 50, 0)
            inpatient = st.number_input("Prior inpatient visits", 0, 50, 0)
            diagnoses = st.number_input("Number of diagnoses", 1, 16, 9)

        with right:
            st.markdown("**Clinical**")
            diag_1 = st.text_input("Primary ICD-9 diagnosis", "250")
            diag_2 = st.text_input("Secondary ICD-9 diagnosis", "250")
            diag_3 = st.text_input("Additional ICD-9 diagnosis", "250")
            glucose = st.selectbox("Max glucose serum", ["None", "Norm", ">200", ">300"])
            a1c = st.selectbox("HbA1c result", ["None", "Norm", ">7", ">8"])
            insulin_status = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
            metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
            change = st.selectbox("Medication change", ["No", "Ch"])
            diabetes_med = st.selectbox("Diabetes medication", ["Yes", "No"])

        submitted = st.form_submit_button("Score patient", type="primary", use_container_width=True)

    if not submitted:
        return

    record = dict(config.DEFAULTS)
    record.update(
        {
            "age": age,
            "gender": gender,
            "race": race,
            "admission_type_id": admission_type,
            "discharge_disposition_id": discharge,
            "admission_source_id": source,
            "medical_specialty": specialty or "Unknown",
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": lab_procedures,
            "num_procedures": procedures,
            "num_medications": medications,
            "number_outpatient": outpatient,
            "number_emergency": emergency,
            "number_inpatient": inpatient,
            "number_diagnoses": diagnoses,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "max_glu_serum": glucose,
            "A1Cresult": a1c,
            "insulin": insulin_status,
            "metformin": metformin,
            "change": change,
            "diabetesMed": diabetes_med,
        }
    )

    try:
        result, engine = predict_record(record)
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
        return

    probability = float(result["probability"])
    threshold = float(result.get("threshold", metadata.get("threshold", 0.5)))
    level = result["risk_level"]

    left, right = st.columns([1, 1.4])
    with left:
        st.metric("30-day readmission probability", f"{100 * probability:.1f}%")
        st.markdown(
            f'<span class="risk-badge" style="background:{risk_color(level)}">'
            f"{level.upper()} RISK</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.write(f"Decision: **{'Readmission likely' if result['prediction'] else 'No readmission predicted'}**")
        st.write(f"Threshold: `{threshold:.2f}`")
        st.caption(f"Scored by {engine}.")

    with right:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=100 * probability,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color(level)},
                    "steps": [
                        {"range": [0, 100 * threshold], "color": "#d1fae5"},
                        {"range": [100 * threshold, min(200 * threshold, 95)], "color": "#fef3c7"},
                        {"range": [min(200 * threshold, 95), 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f172a", "width": 3},
                        "thickness": 0.75,
                        "value": 100 * threshold,
                    },
                },
                title={"text": "Risk gauge"},
            )
        )
        gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)


def render_batch() -> None:
    st.title("Batch Scoring")
    st.caption(
        "Upload a CSV with raw encounter columns (same schema as `diabetic_data.csv`). "
        f"Scoring source: {'FastAPI' if api_online() else 'local artifact'}."
    )

    uploaded = st.file_uploader("Encounter CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        frame = pd.read_csv(uploaded, keep_default_na=False)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.write(f"Loaded **{len(frame):,}** rows and **{len(frame.columns)}** columns.")
    st.dataframe(frame.head(10), use_container_width=True, hide_index=True)

    if not st.button("Score batch", type="primary"):
        return

    try:
        scored, engine = predict_frame(frame)
    except Exception as exc:
        st.error(f"Batch scoring failed: {exc}")
        return

    st.success(f"Scored {len(scored):,} rows with {engine}.")
    left, middle, right = st.columns(3)
    left.metric("Mean probability", f"{100 * scored['probability'].mean():.1f}%")
    middle.metric("Predicted high risk", int((scored["risk_level"] == "high").sum()))
    right.metric("Predicted readmissions", int(scored["prediction"].sum()))

    st.dataframe(scored, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="readmission_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    metadata = load_json(str(config.MODELS_DIR / config.METADATA_FILE))
    eda = load_json(str(config.METRICS_DIR / "eda_summary.json"))
    comparison = load_comparison()

    st.sidebar.title("Readmission Analytics")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Data Insights", "Model Performance", "Predict", "Batch Predict"],
    )
    st.sidebar.divider()
    st.sidebar.caption(f"API endpoint: `{API_URL}`")
    st.sidebar.caption(f"API status: {'online' if api_online() else 'offline (local fallback)'}")
    if metadata:
        st.sidebar.caption(f"Model: `{metadata.get('model_name')}`")
    st.sidebar.divider()
    st.sidebar.caption(
        "Dataset: Diabetes 130-US Hospitals (UCI, CC BY 4.0). "
        "For research and demonstration only — not clinical advice."
    )

    if page == "Overview":
        render_overview(metadata, eda, comparison)
    elif page == "Data Insights":
        render_data_insights(eda)
    elif page == "Model Performance":
        render_model_performance(metadata, comparison)
    elif page == "Predict":
        render_predict(metadata)
    else:
        render_batch()


if __name__ == "__main__":
    main()
