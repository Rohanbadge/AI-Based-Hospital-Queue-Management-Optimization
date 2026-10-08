# 🏥 AI-Based Hospital Queue Management Optimization

An intelligent machine learning solution designed to optimize hospital patient queueing, triage, and waiting time management. This system dynamically evaluates patient urgency, compares algorithmic triage against traditional manual practices, and provides real-time wait time predictions to streamline hospital operations.

---

## 📌 Features

- **Automated Medical Triage:** Evaluates incoming patient symptoms and vital signs to assign appropriate priority scores.
- **Queue Optimization Algorithm:** Dynamically re-orders patient queues based on emergency urgency, expected consultation duration, and resource availability.
- **Model Evaluation & Feature Comparison:** Includes comparison modules (`compare_manual_vs_algorithmic_features.py`) to analyze performance improvements over manual triage methods.
- **Synthetic Dataset Generation:** Built-in tools to generate synthetic medical triage data (`synthetic_medical_triage.csv`) for testing and benchmarking.
- **Modular Backend:** Clean code structure separating training pipelines, evaluation logic, and utility functions.

---

## 📁 Repository Structure

```text
hospital-queue-optimization/
├── Dataset/
│   └── synthetic_medical_triage.csv        # Triage and patient queue dataset
├── backend/
│   ├── training/
│   │   ├── compare_manual_vs_algorithmic_features.py  # Feature comparison script
│   │   ├── evaluate_model.py                         # Model performance metrics
│   │   └── show_model_comparison.py                  # Visualization & comparison logic
│   └── utils/
│       └── utils.py                                  # Helper and utility functions
├── requirements.txt                         # Project dependencies
└── README.md                                # Project documentation
