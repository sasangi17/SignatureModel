from pathlib import Path
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "dataset_ready"

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "best_signature_model.keras"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

CLASS_NAMES = [
    "not_signature",
    "signature"
]

# ==========================================
# LOAD TEST DATA
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(

    DATASET_DIR / "test",

    class_names=CLASS_NAMES,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False
)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ==========================================
# PREDICT
# ==========================================

print("Evaluating test dataset...")

y_true = []

y_probabilities = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy().flatten()
    )

    y_probabilities.extend(
        predictions.flatten()
    )

y_true = np.array(
    y_true
).astype(int)

y_probabilities = np.array(
    y_probabilities
)

# Threshold: 0.5 initially
y_pred = (
    y_probabilities >= 0.5
).astype(int)

# ==========================================
# RESULTS
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\nAccuracy:")
print(
    f"{accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

matrix = confusion_matrix(
    y_true,
    y_pred
)

print(matrix)

# ==========================================
# SAVE RESULTS
# ==========================================

results_path = (
    BASE_DIR
    / "outputs"
    / "evaluation_results.txt"
)

with open(
    results_path,
    "w"
) as file:

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(matrix)
    )

print(
    "\nResults saved:",
    results_path
)
