from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.utils.ml_utils import (
    AUTOENCODER_PATH,
    MODEL_DIR,
    RF_MODEL_PATH,
    build_rf_frame,
    fit_preprocessors,
    load_dataset,
    save_preprocessor_bundle,
    transform_for_autoencoder,
)


SEED = 42
METRICS_PATH = MODEL_DIR / "training_metrics.json"


def build_candidate_models() -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=2,
            random_state=SEED,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=1,
        ),
    }


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)


def build_autoencoder(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(24, activation="relu")(inputs)
    encoded = layers.Dense(12, activation="relu")(encoded)
    bottleneck = layers.Dense(6, activation="relu")(encoded)
    decoded = layers.Dense(12, activation="relu")(bottleneck)
    decoded = layers.Dense(24, activation="relu")(decoded)
    outputs = layers.Dense(input_dim, activation="linear")(decoded)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def compute_reconstruction_error(model: keras.Model, matrix: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(matrix, verbose=0)
    return np.mean(np.square(matrix - reconstructed), axis=1)


def evaluate_classifier(name: str, model, x_train, x_test, y_train, y_test) -> tuple[dict, object]:
    fit_kwargs = {}
    if name == "xgboost":
        fit_kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train)

    model.fit(x_train, y_train, **fit_kwargs)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }
    return metrics, model


def select_training_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    k_features: int = 9,
) -> tuple[pd.DataFrame, pd.DataFrame, SelectKBest, list[str]]:
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k_features, x_train.shape[1]))
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)
    selected_columns = x_train.columns[selector.get_support()].tolist()

    return (
        pd.DataFrame(x_train_selected, columns=selected_columns, index=x_train.index),
        pd.DataFrame(x_test_selected, columns=selected_columns, index=x_test.index),
        selector,
        selected_columns,
    )


def train_models() -> dict:
    set_seeds()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    preprocessors = fit_preprocessors(dataset)
    save_preprocessor_bundle(preprocessors)

    normal_cases = dataset[dataset["triage_level"].isin([0, 1])].copy()
    normal_matrix = transform_for_autoencoder(normal_cases, preprocessors)

    train_matrix, validation_matrix = train_test_split(
        normal_matrix,
        test_size=0.2,
        random_state=SEED,
    )

    autoencoder = build_autoencoder(train_matrix.shape[1])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]

    history = autoencoder.fit(
        train_matrix,
        train_matrix,
        validation_data=(validation_matrix, validation_matrix),
        epochs=60,
        batch_size=64,
        verbose=0,
        callbacks=callbacks,
    )

    full_matrix = transform_for_autoencoder(dataset, preprocessors)
    anomaly_scores = compute_reconstruction_error(autoencoder, full_matrix)

    rf_input = build_rf_frame(dataset, anomaly_scores, preprocessors)
    x_train, x_test, y_train, y_test = train_test_split(
        rf_input,
        dataset["triage_level"],
        test_size=0.2,
        random_state=SEED,
        stratify=dataset["triage_level"],
    )
    x_train_selected, x_test_selected, selector, selected_features = select_training_features(
        x_train,
        x_test,
        y_train,
    )

    comparison_metrics = {}
    trained_models = {}
    for model_name, model in build_candidate_models().items():
        model_metrics, trained_model = evaluate_classifier(
            model_name,
            model,
            x_train_selected,
            x_test_selected,
            y_train,
            y_test,
        )
        comparison_metrics[model_name] = model_metrics
        trained_models[model_name] = trained_model

    best_model_name = max(
        comparison_metrics,
        key=lambda name: (
            comparison_metrics[name]["weighted_f1"],
            comparison_metrics[name]["accuracy"],
            comparison_metrics[name]["macro_f1"],
        ),
    )
    best_model = trained_models[best_model_name]

    autoencoder.save(AUTOENCODER_PATH)
    joblib.dump(best_model, RF_MODEL_PATH)

    threshold = float(np.percentile(compute_reconstruction_error(autoencoder, validation_matrix), 95))
    preprocessors["anomaly_threshold"] = threshold
    preprocessors["feature_selector"] = selector
    preprocessors["selected_rf_features"] = selected_features
    save_preprocessor_bundle(preprocessors)

    metrics = {
        "dataset_rows": int(len(dataset)),
        "normal_cases": int(len(normal_cases)),
        "autoencoder_best_val_loss": float(min(history.history["val_loss"])),
        "selected_model": best_model_name,
        "selected_rf_features": selected_features,
        "anomaly_threshold": round(threshold, 6),
        "model_comparison": comparison_metrics,
        "classification_report": comparison_metrics[best_model_name]["classification_report"],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train_models()
    print(json.dumps(result, indent=2))
