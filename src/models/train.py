"""Train, compare and persist readmission prediction models."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.data.download import download_dataset, load_dataset, update_provenance
from src.features.build import build_features, build_preprocessor, get_feature_names
from src.models.evaluate import (
    calibration_data,
    compute_metrics,
    curve_data,
    tune_threshold,
)

LOGGER = logging.getLogger("readmission.train")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _target(frame: pd.DataFrame) -> pd.Series:
    return (frame[config.TARGET_COLUMN].astype(str).str.strip() == "<30").astype(int)


def _split_data(frame: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    folds = list(splitter.split(frame, y, groups=frame[config.GROUP_COLUMN]))
    test_index = folds[0][1]
    validation_index = folds[1][1]
    train_index = np.setdiff1d(
        np.arange(len(frame)),
        np.concatenate([test_index, validation_index]),
    )

    groups = frame[config.GROUP_COLUMN]
    train_groups = set(groups.iloc[train_index])
    validation_groups = set(groups.iloc[validation_index])
    test_groups = set(groups.iloc[test_index])
    assert train_groups.isdisjoint(validation_groups), "Patient leakage between train/validation"
    assert train_groups.isdisjoint(test_groups), "Patient leakage between train/test"
    assert validation_groups.isdisjoint(test_groups), "Patient leakage between validation/test"
    return train_index, validation_index, test_index


def _sklearn_models(scale_pos_weight: float) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=config.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=config.RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def _train_keras(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    class_weight: dict[int, float],
) -> tuple[object, object]:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(config.RANDOM_STATE)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x_train.shape[1],)),
            keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="readmission_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=60,
        batch_size=512,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    return model, history


def _permutation_importance(
    predict_fn,
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Model-agnostic permutation importance using ROC-AUC loss."""
    generator = np.random.default_rng(random_state)
    baseline = float(roc_auc_score(target, predict_fn(matrix)))
    importances = np.zeros(matrix.shape[1], dtype=float)

    for column in range(matrix.shape[1]):
        shuffled_scores = []
        for _ in range(n_repeats):
            permuted = matrix.copy()
            generator.shuffle(permuted[:, column])
            shuffled_scores.append(float(roc_auc_score(target, predict_fn(permuted))))
        importances[column] = baseline - float(np.mean(shuffled_scores))

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _plot_model_diagnostics(results: dict, best_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axis = plt.subplots(figsize=(8, 6))
    for name, result in results.items():
        axis.plot(
            result["curves"]["roc"]["fpr"],
            result["curves"]["roc"]["tpr"],
            label=f"{name} (AUC={result['test']['roc_auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves — test set")
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for name, result in results.items():
        axis.plot(
            result["curves"]["pr"]["recall"],
            result["curves"]["pr"]["precision"],
            label=f"{name} (PR-AUC={result['test']['pr_auc']:.3f})",
        )
    baseline = results[best_name]["test"]["n_positive"] / results[best_name]["test"]["n_samples"]
    axis.axhline(baseline, linestyle="--", color="grey", label=f"Baseline ({baseline:.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves — test set")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)

    comparison = pd.DataFrame(
        {
            name: {
                "PR-AUC": result["test"]["pr_auc"],
                "ROC-AUC": result["test"]["roc_auc"],
                "F1": result["test"]["f1"],
                "Recall": result["test"]["recall"],
                "Precision": result["test"]["precision"],
            }
            for name, result in results.items()
        }
    ).T
    comparison = comparison.sort_values("PR-AUC", ascending=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    comparison[["PR-AUC", "ROC-AUC", "F1", "Recall"]].plot(kind="bar", ax=axis)
    axis.set_title("Model comparison — test set")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    best_result = results[best_name]
    matrix = np.array(
        [
            [best_result["test"]["confusion_matrix"]["true_negative"], best_result["test"]["confusion_matrix"]["false_positive"]],
            [best_result["test"]["confusion_matrix"]["false_negative"], best_result["test"]["confusion_matrix"]["true_positive"]],
        ]
    )
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, f"{value:,}", ha="center", va="center", fontsize=12)
    axis.set_xticks([0, 1], ["Predicted no", "Predicted yes"])
    axis.set_yticks([0, 1], ["Actual no", "Actual yes"])
    axis.set_title(f"Confusion matrix — {best_name}")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix_best.png", dpi=150)
    plt.close(fig)

    calibration = best_result["calibration"]
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    axis.plot(
        calibration["mean_predicted"],
        calibration["fraction_positive"],
        marker="o",
        label=best_name,
    )
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive fraction")
    axis.set_title(f"Calibration — {best_name}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "calibration_best.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(best_name: str, importance_frame: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance_frame.head(20).iloc[::-1]
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(top["feature"], top["importance"], color="#0f766e")
    axis.set_xlabel("Mean decrease in ROC-AUC")
    axis.set_title(f"Permutation importance — {best_name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance_best.png", dpi=150)
    plt.close(fig)


def run_training(skip_keras: bool = False, quick: bool = False) -> dict:
    """Execute the full training and evaluation workflow."""
    _configure_logging()
    config.ensure_directories()

    csv_path = download_dataset()
    frame = load_dataset(csv_path)
    update_provenance(rows=len(frame), columns=len(frame.columns))
    LOGGER.info("Loaded dataset: %s rows x %s columns", *frame.shape)

    if quick:
        frame = frame.sample(n=min(30_000, len(frame)), random_state=config.RANDOM_STATE)
        frame = frame.reset_index(drop=True)
        LOGGER.info("Quick mode enabled: %s rows", len(frame))

    target = _target(frame)
    LOGGER.info(
        "30-day readmission rate: %.2f%% (%s positives)",
        100 * target.mean(),
        int(target.sum()),
    )

    train_index, validation_index, test_index = _split_data(frame, target)
    LOGGER.info(
        "Split sizes — train: %s, validation: %s, test: %s",
        len(train_index),
        len(validation_index),
        len(test_index),
    )

    features = build_features(frame)
    x_train = features.iloc[train_index]
    x_validation = features.iloc[validation_index]
    x_test = features.iloc[test_index]
    y_train = target.iloc[train_index].to_numpy()
    y_validation = target.iloc[validation_index].to_numpy()
    y_test = target.iloc[test_index].to_numpy()

    split_frame = frame.copy()
    split_frame["target"] = target.to_numpy()
    split_frame["split"] = "train"
    split_frame.loc[split_frame.index[validation_index], "split"] = "validation"
    split_frame.loc[split_frame.index[test_index], "split"] = "test"
    split_frame.to_parquet(config.PROCESSED_DIR / "modeling_dataset.parquet", index=False)

    preprocessor = build_preprocessor()
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_validation_transformed = preprocessor.transform(x_validation)
    x_test_transformed = preprocessor.transform(x_test)
    joblib.dump(preprocessor, config.MODELS_DIR / config.PREPROCESSOR_FILE, compress=3)
    feature_names = get_feature_names(preprocessor)
    LOGGER.info("Transformed feature matrix: %s", x_train_transformed.shape)

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / max(positives, 1)
    class_weight = {
        0: len(y_train) / (2 * max(negatives, 1)),
        1: len(y_train) / (2 * max(positives, 1)),
    }

    results: dict[str, dict] = {}
    sklearn_pipelines: dict[str, Pipeline] = {}

    for name, estimator in _sklearn_models(scale_pos_weight).items():
        LOGGER.info("Training %s", name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started

        validation_probability = pipeline.predict_proba(x_validation)[:, 1]
        test_probability = pipeline.predict_proba(x_test)[:, 1]
        threshold, _ = tune_threshold(y_validation, validation_probability)

        results[name] = {
            "threshold": threshold,
            "train_seconds": round(train_seconds, 2),
            "validation": compute_metrics(y_validation, validation_probability, threshold),
            "test": compute_metrics(y_test, test_probability, threshold),
            "curves": curve_data(y_test, test_probability),
            "calibration": calibration_data(y_test, test_probability),
        }
        sklearn_pipelines[name] = pipeline
        LOGGER.info(
            "%-24s val PR-AUC=%.4f | test PR-AUC=%.4f | test F1=%.4f | test recall=%.4f",
            name,
            results[name]["validation"]["pr_auc"],
            results[name]["test"]["pr_auc"],
            results[name]["test"]["f1"],
            results[name]["test"]["recall"],
        )

    keras_model = None
    keras_history = None
    if not skip_keras:
        LOGGER.info("Training keras_mlp")
        try:
            keras_model, keras_history = _train_keras(
                x_train_transformed,
                y_train,
                x_validation_transformed,
                y_validation,
                class_weight,
            )
            validation_probability = keras_model.predict(x_validation_transformed, verbose=0).ravel()
            test_probability = keras_model.predict(x_test_transformed, verbose=0).ravel()
            threshold, _ = tune_threshold(y_validation, validation_probability)
            results["keras_mlp"] = {
                "threshold": threshold,
                "train_seconds": None,
                "validation": compute_metrics(y_validation, validation_probability, threshold),
                "test": compute_metrics(y_test, test_probability, threshold),
                "curves": curve_data(y_test, test_probability),
                "calibration": calibration_data(y_test, test_probability),
            }
            keras_model.save(config.MODELS_DIR / config.KERAS_MODEL_FILE)
            LOGGER.info(
                "%-24s val PR-AUC=%.4f | test PR-AUC=%.4f | test F1=%.4f | test recall=%.4f",
                "keras_mlp",
                results["keras_mlp"]["validation"]["pr_auc"],
                results["keras_mlp"]["test"]["pr_auc"],
                results["keras_mlp"]["test"]["f1"],
                results["keras_mlp"]["test"]["recall"],
            )

            if keras_history is not None:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                history = keras_history.history
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(history["loss"], label="train")
                axes[0].plot(history["val_loss"], label="validation")
                axes[0].set_title("Keras loss")
                axes[0].set_xlabel("Epoch")
                axes[0].legend()
                axes[1].plot(history["auc"], label="train")
                axes[1].plot(history["val_auc"], label="validation")
                axes[1].set_title("Keras ROC-AUC")
                axes[1].set_xlabel("Epoch")
                axes[1].legend()
                fig.tight_layout()
                fig.savefig(config.FIGURES_DIR / "keras_training_curves.png", dpi=150)
                plt.close(fig)
        except Exception as exc:  # pragma: no cover - environment dependent
            LOGGER.exception("Keras training failed, continuing with scikit-learn models: %s", exc)

    best_name = max(
        results,
        key=lambda name: (results[name]["validation"]["pr_auc"], results[name]["validation"]["f1"]),
    )
    best_result = results[best_name]
    LOGGER.info("Selected best model: %s", best_name)

    if best_name == "keras_mlp":
        best_model_type = "keras"
        best_pipeline = sklearn_pipelines.get(
            max(
                sklearn_pipelines,
                key=lambda name: results[name]["validation"]["pr_auc"],
            )
        )
        joblib.dump(best_pipeline, config.MODELS_DIR / "best_sklearn_model.joblib", compress=3)
    else:
        best_model_type = "sklearn"
        best_pipeline = sklearn_pipelines[best_name]
        joblib.dump(best_pipeline, config.MODELS_DIR / config.BEST_MODEL_FILE, compress=3)

    if best_model_type == "sklearn":
        sklearn_model = best_pipeline.named_steps["model"]

        def predict_fn(matrix):
            return sklearn_model.predict_proba(matrix)[:, 1]

    else:

        def predict_fn(matrix):
            return keras_model.predict(matrix, verbose=0).ravel()

    sample_size = min(3000, len(x_test_transformed))
    importance_frame = _permutation_importance(
        predict_fn,
        x_test_transformed[:sample_size],
        y_test[:sample_size],
        feature_names,
    )
    importance_frame.to_csv(config.METRICS_DIR / "feature_importance.csv", index=False)
    _plot_feature_importance(best_name, importance_frame)
    _plot_model_diagnostics(results, best_name)

    comparison_rows = []
    for name, result in results.items():
        row = {
            "model": name,
            "threshold": result["threshold"],
            "train_seconds": result["train_seconds"],
        }
        for split_name in ("validation", "test"):
            for metric, value in result[split_name].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[f"{split_name}_{metric}"] = value
        comparison_rows.append(row)
    comparison_frame = pd.DataFrame(comparison_rows).sort_values(
        "validation_pr_auc", ascending=False
    )
    comparison_frame.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    metrics_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": config.DATASET_NAME,
            "doi": config.DATASET_DOI,
            "license": config.DATASET_LICENSE,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "positive_rate": float(target.mean()),
        },
        "splits": {
            "train": int(len(train_index)),
            "validation": int(len(validation_index)),
            "test": int(len(test_index)),
            "strategy": "StratifiedGroupKFold on patient_nbr (no patient leakage)",
        },
        "best_model": {
            "name": best_name,
            "model_type": best_model_type,
            "threshold": best_result["threshold"],
        },
        "models": results,
    }
    (config.METRICS_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    metadata = {
        "model_name": best_name,
        "model_type": best_model_type,
        "threshold": best_result["threshold"],
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "dataset": metrics_payload["dataset"],
        "splits": metrics_payload["splits"],
        "metrics": {
            "validation": best_result["validation"],
            "test": best_result["test"],
        },
        "all_models": {
            name: {
                "validation": result["validation"],
                "test": result["test"],
                "threshold": result["threshold"],
            }
            for name, result in results.items()
        },
        "features": {
            "numeric": config.MODEL_NUMERIC,
            "categorical": config.MODEL_CATEGORICAL,
            "n_transformed": int(x_train_transformed.shape[1]),
            "names": feature_names,
        },
        "artifacts": {
            "preprocessor": config.PREPROCESSOR_FILE,
            "best_model": config.BEST_MODEL_FILE if best_model_type == "sklearn" else None,
            "keras_model": config.KERAS_MODEL_FILE if keras_model is not None else None,
            "fallback_sklearn": "best_sklearn_model.joblib"
            if (config.MODELS_DIR / "best_sklearn_model.joblib").exists()
            else None,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": __import__("xgboost").__version__,
            "tensorflow": __import__("tensorflow").__version__ if not skip_keras else None,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (config.MODELS_DIR / config.METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, default=_json_default),
        encoding="utf-8",
    )

    LOGGER.info("Artifacts saved to %s", config.MODELS_DIR)
    LOGGER.info("\n%s", comparison_frame.to_string(index=False))
    return {
        "best_model": best_name,
        "best_model_type": best_model_type,
        "metrics": metadata["metrics"],
        "comparison": comparison_frame,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train readmission prediction models.")
    parser.add_argument("--skip-keras", action="store_true", help="Skip the TensorFlow MLP.")
    parser.add_argument("--quick", action="store_true", help="Train on a 30k-row sample.")
    arguments = parser.parse_args(argv)
    run_training(skip_keras=arguments.skip_keras, quick=arguments.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
