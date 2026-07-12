"""
=========================================================
External Evaluation - MobileNetV2
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

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

MODEL_PATH = Path(
    "models/trained/mobilenet.keras"
)

DATASET_PATH = Path(
    "datasets/resized_external"
)

# ----------------------------------------------------
# External Dataset
# ----------------------------------------------------

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = test_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=(224, 224),

    batch_size=16,

    class_mode=None,

    shuffle=False
)

print()
print("=" * 70)
print("Loading MobileNet Model...")
print("=" * 70)

model = load_model(MODEL_PATH)

print()
print("=" * 70)
print("Generating Predictions...")
print("=" * 70)

predictions = model.predict(
    test_data
)

y_pred_full = np.argmax(
    predictions,
    axis=1
)

# ----------------------------------------------------
# External dataset only has 6 classes
# ----------------------------------------------------

external_classes = [

    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Tomato_Fresh",
    "Tomato_Rotten"
]

# Mapping from MobileNet 8 classes

model_classes = [

    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten",
    "Tomato_Fresh",
    "Tomato_Rotten"
]

model_index_to_external = {

    0: 0,
    1: 1,
    2: 2,
    3: 3,
    6: 4,
    7: 5
}

# Ignore strawberry predictions

y_pred = []

for pred in y_pred_full:

    if pred in model_index_to_external:

        y_pred.append(
            model_index_to_external[pred]
        )

    else:

        # Strawberry predicted
        # Assign nearest tomato class

        y_pred.append(5)

y_pred = np.array(y_pred)

y_true = test_data.classes

# ----------------------------------------------------
# Metrics
# ----------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

print()
print("=" * 70)
print("External Evaluation")
print("=" * 70)

print(
    f"External Accuracy : {accuracy:.4f}"
)

print()
print("=" * 70)
print("Classification Report")
print("=" * 70)

print(

    classification_report(

        y_true,
        y_pred,

        target_names=external_classes,

        digits=4,

        zero_division=0
    )
)

# ----------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10, 8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=external_classes,

    yticklabels=external_classes
)

plt.title(
    "MobileNet External Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(

    "outputs/plots/mobilenet_external_confusion_matrix.png",

    dpi=300,

    bbox_inches="tight"
)

plt.show()

print()
print("=" * 70)
print("External Evaluation Completed")
print("=" * 70)
print(
    "Confusion Matrix Saved."
)

