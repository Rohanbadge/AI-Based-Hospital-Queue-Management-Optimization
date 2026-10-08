from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.training.train_models import METRICS_PATH, train_models


def load_or_generate_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return train_models()


def print_comparison(metrics: dict) -> None:
    comparison = metrics.get("model_comparison", {})
    if not comparison:
        raise ValueError("No model comparison data found in metrics.")

    rows = []
    for model_name, values in comparison.items():
        rows.append(
            (
                model_name,
                float(values["accuracy"]),
                float(values["weighted_f1"]),
                float(values["macro_f1"]),
            )
        )

    ranked = sorted(rows, key=lambda row: (row[2], row[1], row[3]), reverse=True)

    print("=== Model Evaluation Comparison ===")
    print(f"Dataset rows: {metrics.get('dataset_rows', 'N/A')}")
    print(f"Selected best model: {metrics.get('selected_model', 'N/A')}")
    print()
    print(f"{'Rank':<6}{'Model':<20}{'Accuracy':<12}{'Weighted F1':<14}{'Macro F1':<10}")
    print("-" * 62)
    for idx, (name, accuracy, weighted_f1, macro_f1) in enumerate(ranked, start=1):
        print(f"{idx:<6}{name:<20}{accuracy:<12.4f}{weighted_f1:<14.4f}{macro_f1:<10.4f}")

    print()
    print("CSV Output:")
    print("rank,model,accuracy,weighted_f1,macro_f1")
    for idx, (name, accuracy, weighted_f1, macro_f1) in enumerate(ranked, start=1):
        print(f"{idx},{name},{accuracy:.4f},{weighted_f1:.4f},{macro_f1:.4f}")


if __name__ == "__main__":
    all_metrics = load_or_generate_metrics()
    print_comparison(all_metrics)
