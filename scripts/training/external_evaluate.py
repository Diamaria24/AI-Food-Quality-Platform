"""
=========================================================
External Dataset Evaluation
=========================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = Path(
    "models/trained/baseline_cnn.keras"
)

DATASET_PATH = Path(
    "datasets/resized_external"
)

# =========================================================
# LOAD DATASET
# =========================================================

datagen = ImageDataGenerator(
    rescale=1./255
)

test_data = datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(128,128),

    batch_size=16,

    class_mode="categorical",

    shuffle=False
)

# =========================================================
# LOAD MODEL
# =========================================================

print()
print("="*70)
print("Loading Model...")
print("="*70)

model = load_model(MODEL_PATH)

# =========================================================
# PREDICTIONS
# =========================================================

print()
print("="*70)
print("Generating Predictions...")
print("="*70)

predictions = model.predict(
    test_data,
    verbose=1
)

# =========================================================
# REMOVE STRAWBERRY OUTPUTS
# =========================================================
# Original Model Classes:
#
# 0 Apple_Fresh
# 1 Apple_Rotten
# 2 Banana_Fresh
# 3 Banana_Rotten
# 4 Strawberry_Fresh
# 5 Strawberry_Rotten
# 6 Tomato_Fresh
# 7 Tomato_Rotten
#
# External dataset contains only:
#
# Apple, Banana, Tomato
#
# Therefore remove indices 4 and 5.

predictions = predictions[
    :,
    [0,1,2,3,6,7]
]

# Renormalize probabilities

predictions = (

    predictions /

    predictions.sum(
        axis=1,
        keepdims=True
    )

)

# =========================================================
# FINAL PREDICTIONS
# =========================================================

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_data.classes

class_names = list(
    test_data.class_indices.keys()
)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print()
print("="*70)
print("External Evaluation")
print("="*70)

print(
    f"External Accuracy : {accuracy:.4f}"
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print("="*70)
print("Classification Report")
print("="*70)

print(

    classification_report(

        y_true,

        y_pred,

        target_names=class_names,

        digits=4

    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(10,8)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names

)

plt.title(
    "External Dataset Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(

    "outputs/plots/external_confusion_matrix.png",

    dpi=300

)

plt.show()

print()
print("="*70)
print("External Evaluation Completed")
print("="*70)

print(
    "Confusion Matrix Saved."
)