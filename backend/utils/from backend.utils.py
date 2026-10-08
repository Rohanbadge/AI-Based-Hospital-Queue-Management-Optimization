from backend.utils.ml_utils import load_dataset, fit_preprocessors, transform_for_autoencoder, build_rf_frame
from backend.training.train_models import select_training_features
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np

SEED = 42

dataset = load_dataset()
print("Cleaned dataset shape:", dataset.shape)

preprocessors = fit_preprocessors(dataset)

normal_cases = dataset[dataset["triage_level"].isin([0, 1])].copy()
normal_matrix = transform_for_autoencoder(normal_cases, preprocessors)
print("Normal cases shape:", normal_cases.shape)
print("Autoencoder input shape:", normal_matrix.shape)

full_matrix = transform_for_autoencoder(dataset, preprocessors)
print("Full autoencoder matrix shape:", full_matrix.shape)

autoencoder = keras.models.load_model("models/autoencoder_model.h5", compile=False)
reconstructed = autoencoder.predict(full_matrix, verbose=0)
anomaly_scores = np.mean(np.square(full_matrix - reconstructed), axis=1)

rf_input = build_rf_frame(dataset, anomaly_scores, preprocessors)
print("RF input shape before feature selection:", rf_input.shape)

x_train, x_test, y_train, y_test = train_test_split(
    rf_input,
    dataset["triage_level"],
    test_size=0.2,
    random_state=SEED,
    stratify=dataset["triage_level"],
)

print("x_train shape before selection:", x_train.shape)
print("x_test shape before selection:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

x_train_selected, x_test_selected, selector, selected_features = select_training_features(
    x_train,
    x_test,
    y_train,
)

print("x_train shape after selection:", x_train_selected.shape)
print("x_test shape after selection:", x_test_selected.shape)
print("Selected features:", selected_features)
