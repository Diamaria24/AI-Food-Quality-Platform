# ================================================================
# PAPAYA HSI FINAL RANDOM FOREST CLASSIFICATION
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "papaya_classification",
    "pca_90pc"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "papaya_classification"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "papaya_train_90pc.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "papaya_test_90pc.csv"
)

METRICS_FILE = os.path.join(
    DATA_DIR,
    "papaya_final_random_forest_metrics.csv"
)

PREDICTIONS_FILE = os.path.join(
    DATA_DIR,
    "papaya_final_random_forest_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    DATA_DIR,
    "papaya_final_random_forest_feature_importance.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_DIR,
    "papaya_final_random_forest_confusion_matrix.png"
)

FEATURE_IMPORTANCE_GRAPH = os.path.join(
    OUTPUT_DIR,
    "papaya_final_random_forest_feature_importance.png"
)

# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PAPAYA HSI FINAL RANDOM FOREST CLASSIFICATION")
print("=" * 70)

# ================================================================
# LOAD DATA
# ================================================================

print("\n" + "=" * 70)
print("LOADING PAPAYA 90-PC CLASSIFICATION DATA")
print("=" * 70)

print("\nTraining file:")
print(TRAIN_FILE)

print("\nTesting file:")
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

print("\nTraining shape :", train_df.shape)
print("Testing shape  :", test_df.shape)

# ================================================================
# DISPLAY ACTUAL COLUMNS
# ================================================================

print("\n" + "=" * 70)
print("INPUT COLUMNS")
print("=" * 70)

print("\nTraining columns:")
print(train_df.columns.tolist())

# ================================================================
# IDENTIFY PCA FEATURES
# ================================================================

pca_features = [
    col
    for col in train_df.columns
    if col.startswith("PC")
]

pca_features = sorted(
    pca_features,
    key=lambda x: int(x.replace("PC", ""))
)

print("\n" + "=" * 70)
print("PCA FEATURES")
print("=" * 70)

print("\nNumber of PCA components:", len(pca_features))

print("\nPCA features:")
print(pca_features)

if len(pca_features) != 90:
    raise ValueError(
        f"Expected 90 PCA components, "
        f"but found {len(pca_features)}."
    )

# ================================================================
# PAPAYA COLUMN STRUCTURE
# ================================================================
#
# The Papaya 90-PC preparation file uses:
#
#   label
#   Sample_ID
#
# instead of Tomato's:
#
#   class
#   source_file
#
# Therefore we use the actual Papaya columns here.
#
# ================================================================

if "label" not in train_df.columns:
    raise ValueError(
        "Training CSV does not contain 'label'."
    )

if "label" not in test_df.columns:
    raise ValueError(
        "Testing CSV does not contain 'label'."
    )

if "Sample_ID" not in train_df.columns:
    raise ValueError(
        "Training CSV does not contain 'Sample_ID'."
    )

if "Sample_ID" not in test_df.columns:
    raise ValueError(
        "Testing CSV does not contain 'Sample_ID'."
    )

print("\nRequired Papaya columns found.")

# ================================================================
# DATA MATRICES
# ================================================================

X_train = train_df[
    pca_features
].to_numpy(
    dtype=np.float32
)

X_test = test_df[
    pca_features
].to_numpy(
    dtype=np.float32
)

y_train = train_df[
    "label"
].to_numpy()

y_test = test_df[
    "label"
].to_numpy()

# Sample IDs
groups_train = train_df[
    "Sample_ID"
].to_numpy()

groups_test = test_df[
    "Sample_ID"
].to_numpy()

print("\n" + "=" * 70)
print("DATA MATRICES")
print("=" * 70)

print("\nTraining matrix :", X_train.shape)
print("Testing matrix  :", X_test.shape)

print("Training labels :", y_train.shape)
print("Testing labels  :", y_test.shape)

# ================================================================
# NUMERICAL VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)

train_nan = np.isnan(X_train).sum()
test_nan = np.isnan(X_test).sum()

train_inf = np.isinf(X_train).sum()
test_inf = np.isinf(X_test).sum()

print("\nTraining NaN values :", train_nan)
print("Testing NaN values  :", test_nan)

print("Training Inf values :", train_inf)
print("Testing Inf values  :", test_inf)

if train_nan > 0:
    raise ValueError(
        "Training data contains NaN values."
    )

if test_nan > 0:
    raise ValueError(
        "Testing data contains NaN values."
    )

if train_inf > 0:
    raise ValueError(
        "Training data contains infinite values."
    )

if test_inf > 0:
    raise ValueError(
        "Testing data contains infinite values."
    )

# ================================================================
# SOURCE SAMPLE LEAKAGE CHECK
# ================================================================

print("\n" + "=" * 70)
print("SOURCE SAMPLE LEAKAGE CHECK")
print("=" * 70)

train_sources = set(
    np.unique(groups_train)
)

test_sources = set(
    np.unique(groups_test)
)

overlap = train_sources.intersection(
    test_sources
)

print("\nTraining Sample_IDs:", len(train_sources))
print("Testing Sample_IDs :", len(test_sources))
print("Overlapping IDs    :", len(overlap))

if len(overlap) > 0:

    print(
        "\nWARNING: SAMPLE_ID LEAKAGE DETECTED!"
    )

    print(
        "Overlapping IDs:",
        sorted(overlap)
    )

    raise ValueError(
        "Training and testing Sample_IDs overlap."
    )

print(
    "\nSource-sample leakage check: PASS"
)

# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

train_distribution = (
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print(train_distribution)

print("\n" + "=" * 70)
print("TESTING CLASS DISTRIBUTION")
print("=" * 70)

test_distribution = (
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)

print(test_distribution)

# ================================================================
# RANDOM FOREST CONFIGURATION
# ================================================================

N_TREES = 200
RANDOM_STATE = 42

print("\n" + "=" * 70)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 70)

print("\nNumber of trees :", N_TREES)
print("Random state    :", RANDOM_STATE)
print("Training samples:", len(X_train))
print("Input features  :", X_train.shape[1])

# ================================================================
# TRAIN MODEL
# ================================================================

model = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nFitting model...")

model.fit(
    X_train,
    y_train
)

print(
    "Random Forest training completed."
)

# ================================================================
# GENERATE PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test
)

print(
    "\nPredictions generated:",
    len(y_pred)
)

# ================================================================
# RESULTS
# ================================================================

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

print("\n" + "=" * 70)
print("FINAL RANDOM FOREST RESULTS")
print("=" * 70)

print(
    f"\nTEST ACCURACY : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"MACRO PRECISION: "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"MACRO RECALL   : "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"MACRO F1       : "
    f"{macro_f1 * 100:.2f}%"
)

# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

# ================================================================
# PER-CLASS PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

class_labels = sorted(
    np.unique(y_test)
)

class_rows = []

for cls in class_labels:

    precision = precision_score(
        y_test,
        y_pred,
        labels=[cls],
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        labels=[cls],
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        labels=[cls],
        average="macro",
        zero_division=0
    )

    support = np.sum(
        y_test == cls
    )

    class_rows.append({
        "class": cls,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support
    })

class_df = pd.DataFrame(
    class_rows
)

print(
    class_df.to_string(
        index=False
    )
)

# ================================================================
# CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

print(cm)

plt.figure(
    figsize=(9, 8)
)

plt.imshow(cm)

plt.title(
    "Papaya HSI Random Forest Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    range(len(class_labels)),
    class_labels
)

plt.yticks(
    range(len(class_labels)),
    class_labels
)

for i in range(
    len(class_labels)
):

    for j in range(
        len(class_labels)
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved confusion matrix:"
)

print(
    CONFUSION_MATRIX_FILE
)

# ================================================================
# PCA FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 70)
print("PCA FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({
    "feature": pca_features,
    "importance":
        model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    importance_df.to_string(
        index=False
    )
)

importance_df.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)

print(
    "\nSaved feature importance:"
)

print(
    FEATURE_IMPORTANCE_FILE
)

# ================================================================
# FEATURE IMPORTANCE GRAPH
# ================================================================

plt.figure(
    figsize=(14, 7)
)

plt.bar(
    importance_df["feature"],
    importance_df["importance"]
)

plt.title(
    "Papaya HSI Random Forest PCA Feature Importance"
)

plt.xlabel(
    "PCA Component"
)

plt.ylabel(
    "Importance"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_GRAPH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved feature importance graph:"
)

print(
    FEATURE_IMPORTANCE_GRAPH
)

# ================================================================
# SAVE PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame({
    "Sample_ID": groups_test,
    "true_label": y_test,
    "predicted_label": y_pred
})

prediction_df["correct"] = (
    prediction_df["true_label"]
    ==
    prediction_df["predicted_label"]
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    "\nSaved predictions:"
)

print(
    PREDICTIONS_FILE
)

# ================================================================
# SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame([{
    "model":
        "Random Forest",

    "pca_components":
        len(pca_features),

    "n_trees":
        N_TREES,

    "random_state":
        RANDOM_STATE,

    "training_samples":
        len(X_train),

    "testing_samples":
        len(X_test),

    "training_sample_ids":
        len(train_sources),

    "testing_sample_ids":
        len(test_sources),

    "sample_id_overlap":
        len(overlap),

    "accuracy":
        accuracy,

    "macro_precision":
        macro_precision,

    "macro_recall":
        macro_recall,

    "macro_f1":
        macro_f1
}])

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)

print(
    "\nSaved metrics:"
)

print(
    METRICS_FILE
)

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("FINAL PAPAYA HSI RANDOM FOREST COMPLETED")
print("=" * 70)

print(
    f"""
Final test accuracy : {accuracy * 100:.2f}%
Macro F1            : {macro_f1 * 100:.2f}%

PCA components      : {len(pca_features)}
Training samples    : {len(X_train)}
Testing samples     : {len(X_test)}

Training Sample_IDs : {len(train_sources)}
Testing Sample_IDs  : {len(test_sources)}
Sample_ID overlap   : {len(overlap)}

Source-sample leakage: PASS

Classification outputs:
{DATA_DIR}

Graphs:
{OUTPUT_DIR}
"""
)

print("=" * 70)