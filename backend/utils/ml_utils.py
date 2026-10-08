from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "Dataset" / "ED_triage.csv"
MODEL_DIR = ROOT_DIR / "models"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
AUTOENCODER_PATH = MODEL_DIR / "autoencoder_model.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

NUMERIC_FEATURES = [
    "age",
    "heart_rate",
    "systolic_blood_pressure",
    "oxygen_saturation",
    "body_temperature",
    "pain_level",
    "chronic_disease_count",
    "previous_er_visits",
]

CATEGORICAL_FEATURES = ["arrival_mode"]
EXPECTED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["triage_level"]
NUMERIC_BOUNDS = {
    "age": (0, 120),
    "heart_rate": (30, 220),
    "systolic_blood_pressure": (70, 250),
    "oxygen_saturation": (50, 100),
    "body_temperature": (30, 45),
    "pain_level": (0, 10),
    "chronic_disease_count": (0, 10),
    "previous_er_visits": (0, 10),
}
DEFAULT_VALUES = {
    "age": 45,
    "heart_rate": 84,
    "systolic_blood_pressure": 114,
    "oxygen_saturation": 96,
    "body_temperature": 37,
    "pain_level": 1,
    "chronic_disease_count": 0,
    "previous_er_visits": 0,
}
SOURCE_TO_ARRIVAL_MODE = {
    0: "walk_in",
    1: "wheelchair",
    2: "ambulance",
}

RF_FEATURES = [
    "age",
    "heart_rate",
    "systolic_blood_pressure",
    "oxygen_saturation",
    "body_temperature",
    "pain_level",
    "chronic_disease_count",
    "previous_er_visits",
    "arrival_mode",
    "anomaly_score",
]


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET_PATH)
    dataset.columns = [column.strip() for column in dataset.columns]

    if set(EXPECTED_COLUMNS).issubset(dataset.columns):
        normalized = dataset[EXPECTED_COLUMNS].copy()
    elif "TriageGrade" in dataset.columns:
        normalized = normalize_ed_triage_dataset(dataset)
    else:
        raise ValueError(f"Unsupported dataset schema in {DATASET_PATH}.")

    return normalized


def normalize_ed_triage_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(
        {
            "age": pd.to_numeric(dataset.get("age"), errors="coerce"),
            "heart_rate": pd.to_numeric(dataset.get("PulseRate"), errors="coerce"),
            "systolic_blood_pressure": pd.to_numeric(dataset.get("BlooddpressurSystol"), errors="coerce"),
            "oxygen_saturation": pd.to_numeric(dataset.get("O2Saturation"), errors="coerce"),
            "body_temperature": pd.to_numeric(dataset.get("Temperature"), errors="coerce"),
            "pain_level": pd.to_numeric(dataset.get("PainGrade"), errors="coerce"),
            "chronic_disease_count": pd.to_numeric(dataset.get("ref_specialist"), errors="coerce"),
            "previous_er_visits": pd.to_numeric(dataset.get("operational_patient"), errors="coerce"),
            "arrival_mode": pd.to_numeric(dataset.get("Source"), errors="coerce").map(SOURCE_TO_ARRIVAL_MODE),
            # TriageGrade is 1-5 with 1 as most urgent; the app uses 0-3 with 3 as most urgent.
            "triage_level": 4 - pd.to_numeric(dataset.get("TriageGrade"), errors="coerce"),
        }
    )

    normalized = normalized.dropna(subset=["age", "triage_level"]).copy()
    normalized["triage_level"] = normalized["triage_level"].clip(lower=0, upper=3).astype(int)
    normalized["arrival_mode"] = normalized["arrival_mode"].fillna("walk_in")

    for column in NUMERIC_FEATURES:
        lower, upper = NUMERIC_BOUNDS[column]
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        fill_value = normalized[column].median()
        if pd.isna(fill_value):
            fill_value = DEFAULT_VALUES[column]
        normalized[column] = normalized[column].fillna(fill_value).clip(lower=lower, upper=upper)

    return normalized[EXPECTED_COLUMNS]


def fit_preprocessors(dataset: pd.DataFrame) -> dict:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_modes = encoder.fit_transform(dataset[CATEGORICAL_FEATURES])

    base_matrix = np.hstack([dataset[NUMERIC_FEATURES].to_numpy(dtype=float), encoded_modes])

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(base_matrix)

    return {
        "encoder": encoder,
        "scaler": scaler,
        "encoded_feature_names": encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist(),
        "autoencoder_features": NUMERIC_FEATURES + encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist(),
    }


def transform_for_autoencoder(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    encoded_modes = bundle["encoder"].transform(frame[CATEGORICAL_FEATURES])
    base_matrix = np.hstack([frame[NUMERIC_FEATURES].to_numpy(dtype=float), encoded_modes])
    return bundle["scaler"].transform(base_matrix)


def build_rf_frame(frame: pd.DataFrame, anomaly_scores: np.ndarray, bundle: dict) -> pd.DataFrame:
    encoded_modes = bundle["encoder"].transform(frame[CATEGORICAL_FEATURES])
    encoded_columns = bundle["encoded_feature_names"]

    rf_frame = pd.concat(
        [
            frame[NUMERIC_FEATURES].reset_index(drop=True),
            pd.DataFrame(encoded_modes, columns=encoded_columns),
            pd.DataFrame({"anomaly_score": anomaly_scores}),
        ],
        axis=1,
    )
    return rf_frame


def select_rf_features(rf_frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    selector = bundle.get("feature_selector")
    selected_columns = bundle.get("selected_rf_features")

    if selector is None or not selected_columns:
        return rf_frame

    transformed = selector.transform(rf_frame)
    return pd.DataFrame(transformed, columns=selected_columns, index=rf_frame.index)


def ensure_numeric_types(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in NUMERIC_FEATURES:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def save_preprocessor_bundle(bundle: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, SCALER_PATH)


def load_preprocessor_bundle() -> dict:
    return joblib.load(SCALER_PATH)
