"""Exploratory data analysis for the readmission dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.data.download import download_dataset, load_dataset, update_provenance

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("crest")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def run_eda() -> dict:
    """Generate EDA figures and a machine-readable summary."""
    config.ensure_directories()
    frame = load_dataset(download_dataset())
    update_provenance(rows=len(frame), columns=len(frame.columns))

    target = (frame[config.TARGET_COLUMN].astype(str).str.strip() == "<30").astype(int)
    age_order = sorted(
        frame["age"].dropna().unique(),
        key=lambda band: config.AGE_MIDPOINTS.get(band, 0),
    )

    fig, axis = plt.subplots(figsize=(6, 4))
    counts = target.value_counts().sort_index()
    axis.bar(["No 30-day readmission", "30-day readmission"], counts.values, color=["#0f766e", "#f59e0b"])
    for index, value in enumerate(counts.values):
        axis.text(index, value, f"{value:,}", ha="center", va="bottom")
    axis.set_ylabel("Encounters")
    axis.set_title("Target distribution")
    _save(fig, "target_distribution.png")

    missing = frame.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(15)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.barh(missing.index[::-1], 100 * missing.values[::-1], color="#b45309")
    axis.set_xlabel("Missing values (%)")
    axis.set_title("Top missing columns")
    _save(fig, "missing_values.png")

    age_rate = (
        frame.assign(target=target)
        .groupby("age", observed=True)["target"]
        .mean()
        .reindex(age_order)
    )
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.bar(age_rate.index, 100 * age_rate.values, color="#0f766e")
    axis.set_xlabel("Age band")
    axis.set_ylabel("30-day readmission rate (%)")
    axis.set_title("Readmission rate by age")
    axis.tick_params(axis="x", rotation=35)
    _save(fig, "readmission_by_age.png")

    time_rate = (
        frame.assign(target=target)
        .groupby("time_in_hospital", observed=True)["target"]
        .mean()
    )
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(time_rate.index, 100 * time_rate.values, marker="o", color="#0f766e")
    axis.set_xlabel("Days in hospital")
    axis.set_ylabel("30-day readmission rate (%)")
    axis.set_title("Readmission rate by length of stay")
    _save(fig, "readmission_by_time_in_hospital.png")

    inpatient_binned = frame["number_inpatient"].clip(upper=3)
    inpatient_rate = (
        frame.assign(target=target, inpatient_binned=inpatient_binned)
        .groupby("inpatient_binned", observed=True)["target"]
        .mean()
    )
    labels = ["0", "1", "2", "3+"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels[: len(inpatient_rate)], 100 * inpatient_rate.values, color="#0f766e")
    axis.set_xlabel("Inpatient visits in prior year")
    axis.set_ylabel("30-day readmission rate (%)")
    axis.set_title("Readmission rate by prior inpatient utilization")
    _save(fig, "readmission_by_prior_inpatient.png")

    insulin_rate = (
        frame.assign(target=target)
        .groupby("insulin", observed=True)["target"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(insulin_rate.index, 100 * insulin_rate.values, color="#0f766e")
    axis.set_xlabel("Insulin status")
    axis.set_ylabel("30-day readmission rate (%)")
    axis.set_title("Readmission rate by insulin regimen")
    _save(fig, "readmission_by_insulin.png")

    numeric = frame[config.NUMERIC_RAW].apply(pd.to_numeric, errors="coerce")
    fig, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="crest", ax=axis, cbar_kws={"shrink": 0.8})
    axis.set_title("Correlation between numeric features")
    _save(fig, "numeric_correlation.png")

    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "unique_patients": int(frame[config.GROUP_COLUMN].nunique()),
        "encounters_per_patient": round(len(frame) / frame[config.GROUP_COLUMN].nunique(), 3),
        "target_counts": {str(key): int(value) for key, value in target.value_counts().sort_index().items()},
        "target_rate": round(float(target.mean()), 4),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_top_10": {column: round(float(value), 4) for column, value in missing.head(10).items()},
        "age_order": age_order,
        "readmission_rate_by_age": {key: round(float(value), 4) for key, value in age_rate.items()},
        "readmission_rate_by_insulin": {
            key: round(float(value), 4) for key, value in insulin_rate.items()
        },
    }
    (config.METRICS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()
