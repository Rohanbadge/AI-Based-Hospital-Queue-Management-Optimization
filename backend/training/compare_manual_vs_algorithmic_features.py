from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.utils.ml_utils import DATASET_PATH, load_dataset

SEED = 42
REPORT_PATH = ROOT_DIR / "models" / "feature_selection_comparison.csv"
DETAILS_PATH = ROOT_DIR / "models" / "feature_selection_details.json"


def build_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=180,
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
    )


def encode_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.copy()
    for column in encoded.columns:
        if encoded[column].dtype == "object":
            encoded[column] = encoded[column].fillna("missing")
            encoded[column] = pd.Categorical(encoded[column]).codes
    return encoded


def evaluate_manual_features() -> dict:
    dataset = load_dataset().copy()
    y = dataset["triage_level"]
    x = dataset.drop(columns=["triage_level"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    x_train = encode_categoricals(x_train)
    x_test = encode_categoricals(x_test)

    model = build_classifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "approach": "manual_curated_features",
        "num_features": int(x.shape[1]),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "selected_features": x.columns.tolist(),
    }


def evaluate_algorithmic_features(k_features: int = 12) -> dict:
    raw = pd.read_csv(DATASET_PATH)
    raw.columns = [column.strip() for column in raw.columns]

    # Keep broad candidate set from raw data while removing direct target and identifier.
    y = (4 - pd.to_numeric(raw["TriageGrade"], errors="coerce")).clip(lower=0, upper=3)
    feature_frame = raw.drop(columns=["TriageGrade", "triage_code"], errors="ignore").copy()

    for column in feature_frame.columns:
        if feature_frame[column].dtype == "object":
            feature_frame[column] = feature_frame[column].fillna("missing")
            feature_frame[column] = pd.Categorical(feature_frame[column]).codes
        else:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

    valid_rows = y.notna()
    y = y[valid_rows].astype(int)
    feature_frame = feature_frame.loc[valid_rows].reset_index(drop=True)
    y = y.reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    x_all = imputer.fit_transform(feature_frame)

    selector = SelectKBest(score_func=mutual_info_classif, k=min(k_features, x_all.shape[1]))
    x_selected = selector.fit_transform(x_all, y)
    selected_mask = selector.get_support()
    selected_columns = feature_frame.columns[selected_mask].tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x_selected,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    model = build_classifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "approach": "algorithmic_selectkbest",
        "num_features": int(x_selected.shape[1]),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "selected_features": selected_columns,
    }


def print_and_save_report(rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda row: (row["weighted_f1"], row["accuracy"], row["macro_f1"]), reverse=True)

    print("=== Manual vs Algorithmic Feature Selection Comparison ===")
    print(f"{'Rank':<6}{'Approach':<30}{'Features':<10}{'Accuracy':<12}{'Weighted F1':<14}{'Macro F1':<10}")
    print("-" * 84)
    for idx, row in enumerate(ranked, start=1):
        print(
            f"{idx:<6}{row['approach']:<30}{row['num_features']:<10}"
            f"{row['accuracy']:<12.4f}{row['weighted_f1']:<14.4f}{row['macro_f1']:<10.4f}"
        )

    print()
    print("CSV Output:")
    print("rank,approach,num_features,accuracy,weighted_f1,macro_f1")
    for idx, row in enumerate(ranked, start=1):
        print(
            f"{idx},{row['approach']},{row['num_features']},"
            f"{row['accuracy']:.4f},{row['weighted_f1']:.4f},{row['macro_f1']:.4f}"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["rank", "approach", "num_features", "accuracy", "weighted_f1", "macro_f1"])
        for idx, row in enumerate(ranked, start=1):
            writer.writerow(
                [idx, row["approach"], row["num_features"], row["accuracy"], row["weighted_f1"], row["macro_f1"]]
            )

    DETAILS_PATH.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    print()
    print(f"Saved CSV report: {REPORT_PATH}")
    print(f"Saved details (with selected columns): {DETAILS_PATH}")


if __name__ == "__main__":
    manual_result = evaluate_manual_features()
    algorithmic_result = evaluate_algorithmic_features(k_features=12)
    print_and_save_report([manual_result, algorithmic_result])
