from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model

from scripts.training.mobilenet_loader import (
    load_datasets
)

print()
print("="*70)
print("Loading Dataset...")
print("="*70)

_, _, test_data = load_datasets()

print()
print("="*70)
print("Loading MobileNet Model...")
print("="*70)

MODEL_PATH = Path(
    "models/trained/mobilenet.keras"
)

model = load_model(MODEL_PATH)

print(model.summary())

print()
print("="*70)
print("Evaluating...")
print("="*70)

loss, accuracy = model.evaluate(
    test_data
)

print()
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

print()
print("="*70)
print("Generating Predictions...")
print("="*70)

predictions = model.predict(
    test_data
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_data.classes

class_names = list(
    test_data.class_indices.keys()
)

print()
print("="*70)
print("Classification Report")
print("="*70)

print(

    classification_report(

        y_true,
        y_pred,
        target_names=class_names
    )
)

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10,8))

sns.heatmap(

    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "MobileNet Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(

    "outputs/plots/mobilenet_confusion_matrix.png",

    dpi=300
)

plt.show()

print()
print("="*70)
print("Evaluation Completed")
print("="*70)

