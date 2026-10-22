"""
test_tomato_final_rf.py

Final Random Forest classification for Tomato HSI.

Uses the validated 20-PC dataset generated from the real
550-band tomato hyperspectral dataset.

IMPORTANT:
- PCA has already been fitted using training cubes only.
- This script does NOT perform PCA again.
- Train/test separation is at source-cube level.
- PC1-PC20 are used as classifier features.
"""

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt


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
    "tomato_final_random_forest_metrics.csv"
)

PREDICTIONS_FILE = os.path.join(
    DATA_DIR,
    "tomato_final_random_forest_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    DATA_DIR,
    "tomato_final_random_forest_feature_importance.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    GRAPH_DIR,
    "tomato_final_random_forest_confusion_matrix.png"
)

FEATURE_IMPORTANCE_GRAPH = os.path.join(
    GRAPH_DIR,
    "tomato_final_random_forest_feature_importance.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 200

PCA_FEATURES = [
    f"PC{i}"
    for i in range(1, 21)
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TOMATO HSI FINAL RANDOM FOREST CLASSIFICATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING 20-PC CLASSIFICATION DATA")
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
# VALIDATE REQUIRED COLUMNS
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
print("Required columns found.")
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
        "NaN values detected in PCA features."
    )

if train_inf > 0 or test_inf > 0:
    raise ValueError(
        "Infinite values detected in PCA features."
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

print("Training source cubes:", len(train_sources))
print("Testing source cubes :", len(test_sources))
print("Overlapping cubes    :", len(overlap))

if overlap:
    print()
    print("OVERLAPPING SOURCE FILES:")
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
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(
    pd.Series(y_train)
    .value_counts()
)


print()
print("=" * 70)
print("TEST CLASS DISTRIBUTION")
print("=" * 70)

print(
    pd.Series(y_test)
    .value_counts()
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 70)

print("Number of trees :", N_ESTIMATORS)
print("Random state    :", RANDOM_STATE)
print("Training samples:", len(X_train))
print("Input features  :", X_train.shape[1])

print()
print("Fitting model...")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Random Forest training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_pred = model.predict(X_test)

print(
    "Predictions generated:",
    len(y_pred)
)


# ============================================================
# PERFORMANCE METRICS
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


print()
print("=" * 70)
print("FINAL RANDOM FOREST RESULTS")
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
# PER-CLASS METRICS
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
    "Tomato HSI - Final Random Forest Confusion Matrix"
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

# Add values to matrix
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
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("PCA FEATURE IMPORTANCE")
print("=" * 70)

feature_importance_df = pd.DataFrame({
    "feature": PCA_FEATURES,
    "importance": model.feature_importances_
})

feature_importance_df = (
    feature_importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    feature_importance_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance_df.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)

print()
print(
    "Saved feature importance:"
)

print(
    FEATURE_IMPORTANCE_FILE
)


# ============================================================
# FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(11, 7)
)

plt.bar(
    feature_importance_df["feature"],
    feature_importance_df["importance"]
)

plt.title(
    "Tomato HSI - Random Forest PCA Feature Importance"
)

plt.xlabel(
    "Principal Component"
)

plt.ylabel(
    "Feature Importance"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_GRAPH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    "Saved feature importance graph:"
)

print(
    FEATURE_IMPORTANCE_GRAPH
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
    "model": "Random Forest",
    "pca_components": 20,
    "variance_retained_percent": 99.7888,
    "n_estimators": N_ESTIMATORS,
    "random_state": RANDOM_STATE,
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "input_features": len(PCA_FEATURES),
    "accuracy_percent": accuracy * 100,
    "macro_precision_percent": macro_precision * 100,
    "macro_recall_percent": macro_recall * 100,
    "macro_f1_percent": macro_f1 * 100,
    "training_source_cubes": len(train_sources),
    "testing_source_cubes": len(test_sources),
    "source_cube_overlap": len(overlap)
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
print("FINAL TOMATO HSI RANDOM FOREST COMPLETED")
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
    "The model was evaluated on completely"
)

print(
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