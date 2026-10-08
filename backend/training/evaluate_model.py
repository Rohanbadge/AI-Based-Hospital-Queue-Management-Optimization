from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.training.train_models import train_models
from backend.utils.ml_utils import (
    AUTOENCODER_PATH,
    RF_MODEL_PATH,
    SCALER_PATH,
    build_rf_frame,
    load_dataset,
    load_preprocessor_bundle,
    select_rf_features,
    transform_for_autoencoder,
)

SEED = 42


def ensure_artifacts() -> None:
    if RF_MODEL_PATH.exists() and AUTOENCODER_PATH.exists() and SCALER_PATH.exists():
        return

    print("Model artifacts missing. Training models first...")
    train_models()


def compute_anomaly_scores(autoencoder: keras.Model, matrix: np.ndarray) -> np.ndarray:
    reconstructed = autoencoder.predict(matrix, verbose=0)
    return np.mean(np.square(matrix - reconstructed), axis=1)


def run_evaluation() -> dict:
    ensure_artifacts()

    dataset = load_dataset().reset_index(drop=True)
    _, test_indices = train_test_split(
        dataset.index.to_numpy(),
        test_size=0.2,
        random_state=SEED,
        stratify=dataset["triage_level"],
    )

    preprocessor_bundle = load_preprocessor_bundle()
    rf_model = joblib.load(RF_MODEL_PATH)
    autoencoder = keras.models.load_model(AUTOENCODER_PATH, compile=False)

    ae_matrix = transform_for_autoencoder(dataset, preprocessor_bundle)
    anomaly_scores = compute_anomaly_scores(autoencoder, ae_matrix)
    rf_frame = build_rf_frame(dataset, anomaly_scores, preprocessor_bundle)
    selected_rf_frame = select_rf_features(rf_frame, preprocessor_bundle)

    y_true = dataset.loc[test_indices, "triage_level"].to_numpy()
    y_pred = rf_model.predict(selected_rf_frame.loc[test_indices])

    labels = [0, 1, 2, 3]
    metrics = {
        "rows": int(len(dataset)),
        "test_rows": int(len(test_indices)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    return metrics


if __name__ == "__main__":
    results = run_evaluation()
    print("=== Hospital Queue Model Evaluation ===")
    print(f"Rows: {results['rows']} | Test rows: {results['test_rows']}")
    print(f"Accuracy: {results['accuracy']}")
    print(f"Weighted F1: {results['weighted_f1']}")
    print(f"Macro F1: {results['macro_f1']}")
    print("Confusion Matrix [labels 0,1,2,3]:")
    for row in results["confusion_matrix"]:
        print(row)
    print("Classification Report:")
    print(json.dumps(results["classification_report"], indent=2))
