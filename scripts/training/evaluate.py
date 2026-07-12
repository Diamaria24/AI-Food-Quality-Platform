"""
=========================================================
Model Evaluation
=========================================================
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model

from scripts.training.data_loader import (
    load_datasets
)

from scripts.config.constants import (
    TRAINED_MODELS,
    PLOTS_FOLDER
)

def evaluate_model():

    print()

    print("=" * 70)
    print("Loading Test Dataset...")
    print("=" * 70)

    _, _, test_generator = load_datasets()

    print()

    print("=" * 70)
    print("Loading Model...")
    print("=" * 70)

    model = load_model(

        TRAINED_MODELS
        /
        "baseline_cnn.keras"

    )

    print(model.summary())

    print()

    print("=" * 70)
    print("Evaluating Model...")
    print("=" * 70)

    loss, accuracy = model.evaluate(
        test_generator,
        verbose=1
    )

    print()

    print(f"Test Loss      : {loss:.4f}")
    print(f"Test Accuracy  : {accuracy:.4f}")

    print()

    print("=" * 70)
    print("Generating Predictions...")
    print("=" * 70)

    predictions = model.predict(
        test_generator,
        verbose=1
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    y_true = (
        test_generator.classes
    )

    class_names = list(
        test_generator.class_indices.keys()
    )

    print()

    print("=" * 70)
    print("Classification Report")
    print("=" * 70)

    report = classification_report(

        y_true,
        y_pred,
        target_names=class_names

    )

    print(report)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,
        display_labels=class_names

    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        "Confusion Matrix"
    )

    PLOTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(

        PLOTS_FOLDER
        /
        "confusion_matrix.png",

        dpi=300,
        bbox_inches="tight"

    )

    plt.show()

    metrics = {

        "test_accuracy":
            float(accuracy),

        "test_loss":
            float(loss)

    }

    with open(

        PLOTS_FOLDER
        /
        "test_metrics.json",

        "w"

    ) as file:

        json.dump(

            metrics,
            file,
            indent=4

        )

    print()

    print("=" * 70)
    print("Evaluation Completed.")
    print("=" * 70)

    print(

        "Confusion Matrix Saved : "

        f"{PLOTS_FOLDER}"

    )

if __name__ == "__main__":

    evaluate_model()

