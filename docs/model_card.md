# Model Card — 30-Day Readmission Prediction

## Model details

| Field | Value |
|---|---|
| Model name | XGBoost classifier (`XGBClassifier`) |
| Version | 1.0.0 |
| Task | Binary classification: readmission within 30 days |
| Target | `readmitted == "<30"` → 1; `">30"`/`"NO"` → 0 |
| Training date | 2026-11-24 (UTC) |
| Artifact | `models/best_model.joblib` (preprocessing + model pipeline) |
| Decision threshold | 0.53 (selected on validation to maximize F1) |
| Framework | scikit-learn 1.9.1, XGBoost 2.1.4, Python 3.12 |
| Inputs | 36 engineered features → 157 transformed model inputs |
| Output | Probability of 30-day readmission; binary label at threshold |

## Intended use

- **Primary use**: research, education and portfolio demonstration of an
  end-to-end ML pipeline for hospital readmission risk.
- **Secondary use**: batch triage analytics, discharge-planning research,
  hypothesis generation.
- **Out of scope**: individual clinical decision-making, medical advice,
  insurance or employment decisions, any use with identifiable patient data
  without appropriate governance.

## Training data

- **Source**: UCI *Diabetes 130-US Hospitals for Years 1999-2008*
  (DOI 10.24432/C5230J, CC BY 4.0).
- **Coverage**: 101,766 inpatient encounters, 71,518 unique patients,
  130 US hospitals, 1999–2008.
- **Positive rate**: 11.16% (11,357 encounters).
- **Split**: patient-aware `StratifiedGroupKFold` (7 folds): 72,690 train /
  14,537 validation / 14,539 test. No patient appears in more than one split.
- **Exclusions**: records are already limited by the dataset authors to diabetic
  inpatient encounters with 1–14 day stays, lab tests and administered
  medications.
- **Dropped fields**: `weight` (96.9% missing), `payer_code` (39.6% missing),
  identifiers.

## Features

- Demographics: race, gender, age band (mapped to midpoint).
- Encounter: admission type/source/discharge disposition, medical specialty,
  length of stay.
- Utilization: lab procedures, other procedures, medications, prior
  outpatient/emergency/inpatient visits, diagnoses count.
- Clinical: ICD-9 diagnoses grouped into nine clinical categories, glucose and
  HbA1c results, insulin regimen, medication change flags.
- Engineered: active/changed medication counts, prior-visit totals and ratios,
  procedures and medications per day, high-utilizer, polypharmacy and prior
  inpatient flags.

Categorical encoding: one-hot with rare-category infrequency threshold (20) and
unseen-category handling. Numeric: median imputation and standard scaling.

## Evaluation

Test set (14,539 encounters, 1,622 positives), threshold 0.53:

| Metric | Value |
|---|---:|
| Accuracy | 0.7361 |
| Precision | 0.2058 |
| Recall | 0.4778 |
| F1 | 0.2877 |
| ROC-AUC | 0.6748 |
| PR-AUC | 0.2345 |
| Brier score | 0.2078 |

Confusion matrix: TN 9,927 · FP 2,990 · FN 847 · TP 775.

Baseline positive prevalence: 0.1116. PR-AUC lift: **2.1×**.

Comparison (test PR-AUC): HistGB 0.2346, Keras MLP 0.2321, RF 0.2147,
Logistic regression 0.2145. XGBoost was selected on validation PR-AUC
(0.2291 vs 0.2285 for HistGB) before test evaluation; the two tree models are
effectively tied.

### Threshold trade-off

The tuned threshold intentionally favors recall over precision. Lowering the
threshold increases sensitivity (more alerts, more false positives); raising it
reduces alert volume. Any operational use must choose a threshold based on the
cost of missed readmissions versus alert fatigue.

## Interpretability

Top permutation importances (mean decrease in ROC-AUC):

1. `number_inpatient` (0.0219)
2. `inpatient_ratio` (0.0138)
3. `discharge_disposition_id_1` (0.0126)
4. `discharge_disposition_id_11` (0.0119)
5. `discharge_disposition_id_5` (0.0098)
6. `number_diagnoses` (0.0080)
7. `prior_visits_total` (0.0074)
8. `medical_specialty_ObstetricsandGynecology` (0.0055)
9. `time_in_hospital` (0.0044)
10. `medical_specialty_Cardiology` (0.0031)

These associations are predictive, not causal.

## Ethical considerations and fairness

- Race, gender and age are included as features. The dataset contains known
  disparities in care access and documentation; the model has **not** been
  audited for demographic parity, equal opportunity or calibration across
  subgroups. A fairness audit is required before any real-world use.
- The dataset is de-identified and public, but readmission risk models can
  amplify structural inequities if used to allocate resources.
- Predictions reflect 1999–2008 US hospital practice and coding conventions.

## Limitations

- No temporal or external validation; random split may overestimate performance
  on future data.
- Class imbalance limits precision (~21% at the selected threshold).
- Missingness is informative; `Unknown` categories and imputation can hide
  process-of-care differences.
- No causal claims: feature importance is predictive, not interventional.
- No uncertainty quantification beyond probability calibration.
- Not a medical device; no regulatory clearance.

## Maintenance

- Re-train when data drift is detected (population, coding, discharge policy).
- Monitor PR-AUC, recall at fixed alert volume, calibration and subgroup metrics.
- Version artifacts (`metadata.json`) and record the training commit.
- Re-run `pytest`, `ruff check .` and the API smoke test after changes.

## Citation

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., &
Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates.
*BioMed Research International*, 2014, 781670.
