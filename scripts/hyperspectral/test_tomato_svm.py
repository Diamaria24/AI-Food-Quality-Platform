"""
test_tomato_svm.py

SVM comparison model for the finalized Tomato HSI dataset.

Uses:
    - 20 PCA components
    - Same train/test split as final Random Forest
    - Same source-cube separation
    - RBF kernel

IMPORTANT:
PCA is NOT performed here.
The PCA transformation was already fitted using training data only.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "tomato_classification",
    "pca_20pc"
)

GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "tomato_classification"
)

os.makedirs(GRAPH_DIR, exist_ok=True)


TRAIN_FILE = os.path.join(
    DATA_DIR,
    "tomato_train_20pc.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "tomato_test_20pc.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

METRICS_FILE = os.path.join(
    DATA_DIR,
    "tomato_svm_metrics.csv"
)

PREDICTIONS_FILE = os.path.join(
    DATA_DIR,
    "tomato_svm_predictions.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    GRAPH_DIR,
    "tomato_svm_confusion_matrix.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

PCA_FEATURES = [
    f"PC{i}"
    for i in range(1, 21)
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TOMATO HSI SVM CLASSIFICATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TOMATO 20-PC DATA")
print("=" * 70)

print()
print("Training file:")
print(TRAIN_FILE)

print()
print("Testing file:")
print(TEST_FILE)


if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Testing file not found:\n{TEST_FILE}"
    )


train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


print()
print("Training shape :", train_df.shape)
print("Testing shape  :", test_df.shape)


# ============================================================
# COLUMN VALIDATION
# ============================================================

print()
print("=" * 70)
print("COLUMN VALIDATION")
print("=" * 70)

required_columns = PCA_FEATURES + [
    "class",
    "source_file"
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Missing training column: {column}"
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Missing testing column: {column}"
        )

print()
print("All required columns found.")

print()
print("PCA features:")
print(PCA_FEATURES)


# ============================================================
# CREATE MATRICES
# ============================================================

X_train = train_df[PCA_FEATURES].to_numpy(
    dtype=np.float32
)

X_test = test_df[PCA_FEATURES].to_numpy(
    dtype=np.float32
)

y_train = train_df["class"].to_numpy()

y_test = test_df["class"].to_numpy()


print()
print("=" * 70)
print("DATA MATRICES")
print("=" * 70)

print("Training matrix :", X_train.shape)
print("Testing matrix  :", X_test.shape)

print("Training labels :", y_train.shape)
print("Testing labels  :", y_test.shape)


# ============================================================
# NUMERICAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)

train_nan = np.isnan(X_train).sum()
test_nan = np.isnan(X_test).sum()

train_inf = np.isinf(X_train).sum()
test_inf = np.isinf(X_test).sum()

print("Training NaN values :", train_nan)
print("Testing NaN values  :", test_nan)

print("Training Inf values :", train_inf)
print("Testing Inf values  :", test_inf)


if train_nan > 0 or test_nan > 0:
    raise ValueError(
        "NaN values detected."
    )

if train_inf > 0 or test_inf > 0:
    raise ValueError(
        "Infinite values detected."
    )


# ============================================================
# SOURCE-CUBE LEAKAGE CHECK
# ============================================================

print()
print("=" * 70)
print("SOURCE-CUBE LEAKAGE CHECK")
print("=" * 70)

train_sources = set(
    train_df["source_file"].astype(str)
)

test_sources = set(
    test_df["source_file"].astype(str)
)

overlap = train_sources.intersection(
    test_sources
)

print(
    "Training source cubes:",
    len(train_sources)
)

print(
    "Testing source cubes :",
    len(test_sources)
)

print(
    "Overlapping cubes    :",
    len(overlap)
)

if overlap:

    print()
    print("OVERLAPPING FILES:")

    for source in sorted(overlap):
        print(" -", source)

    raise ValueError(
        "Source-cube leakage detected!"
    )

print()
print("Source-cube leakage check: PASS")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print()
print("Training:")

print(
    pd.Series(y_train)
    .value_counts()
)

print()
print("Testing:")

print(
    pd.Series(y_test)
    .value_counts()
)


# ============================================================
# STANDARDIZATION
# ============================================================

print()
print("=" * 70)
print("TRAINING-DERIVED STANDARDIZATION")
print("=" * 70)

print(
    "Fitting StandardScaler using training data only..."
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "Training scaling completed."
)

print(
    "Testing data transformed using "
    "training-derived parameters."
)


# ============================================================
# TRAIN SVM
# ============================================================

print()
print("=" * 70)
print("TRAINING SVM")
print("=" * 70)

print("Kernel      : RBF")
print("Random state:", RANDOM_STATE)
print("Training samples:", len(X_train))
print("Input features :", X_train.shape[1])

print()
print("Fitting SVM...")

model = SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
    random_state=RANDOM_STATE
)

model.fit(
    X_train_scaled,
    y_train
)

print(
    "SVM training completed."
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test_scaled
)

print(
    "Predictions generated:",
    len(y_pred)
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("SVM RESULTS")
print("=" * 70)

print()
print(
    f"TEST ACCURACY : {accuracy * 100:.2f}%"
)

print(
    f"MACRO PRECISION: {macro_precision * 100:.2f}%"
)

print(
    f"MACRO RECALL   : {macro_recall * 100:.2f}%"
)

print(
    f"MACRO F1       : {macro_f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

class_names = sorted(
    np.unique(y_test)
)

per_class_rows = []

for class_name in class_names:

    values = report_dict[str(class_name)]

    per_class_rows.append({
        "class": class_name,
        "precision": values["precision"],
        "recall": values["recall"],
        "f1_score": values["f1-score"],
        "support": values["support"]
    })


per_class_df = pd.DataFrame(
    per_class_rows
)


print()
print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

print(
    per_class_df.to_string(
        index=False
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

labels = list(
    model.classes_
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Tomato HSI - SVM Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(labels)),
    labels,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(labels)),
    labels
)

plt.colorbar()

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    "Saved confusion matrix:"
)

print(
    CONFUSION_MATRIX_FILE
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

predictions_df[
    "predicted_class"
] = y_pred

predictions_df[
    "correct_prediction"
] = (
    predictions_df["class"]
    ==
    predictions_df["predicted_class"]
)


predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print()
print(
    "Saved predictions:"
)

print(
    PREDICTIONS_FILE
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "model": "SVM",

    "kernel": "RBF",

    "C": 10.0,

    "gamma": "scale",

    "pca_components": 20,

    "variance_retained_percent": 99.79,

    "training_samples": len(X_train),

    "testing_samples": len(X_test),

    "input_features": len(PCA_FEATURES),

    "accuracy_percent": accuracy * 100,

    "macro_precision_percent":
        macro_precision * 100,

    "macro_recall_percent":
        macro_recall * 100,

    "macro_f1_percent":
        macro_f1 * 100,

    "training_source_cubes":
        len(train_sources),

    "testing_source_cubes":
        len(test_sources),

    "source_cube_overlap":
        len(overlap),

    "random_state":
        RANDOM_STATE
}])


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)

print()
print(
    "Saved metrics:"
)

print(
    METRICS_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TOMATO HSI SVM COMPLETED")
print("=" * 70)

print()
print(
    f"Final test accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1            : {macro_f1 * 100:.2f}%"
)

print(
    "PCA components      : 20"
)

print(
    "Variance retained   : 99.79%"
)

print(
    "Training cubes      : 5"
)

print(
    "Testing cubes       : 5"
)

print(
    "Source overlap      : 0"
)

print()
print(
    "The SVM was evaluated on completely "
    "unseen source cubes."
)

print()
print(
    "Classification outputs:"
)

print(
    DATA_DIR
)

print()
print(
    "Graphs:"
)

print(
    GRAPH_DIR
)

print()
print("=" * 70)