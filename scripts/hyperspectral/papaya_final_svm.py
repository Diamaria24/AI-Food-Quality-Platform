# ================================================================
# PAPAYA HSI FINAL SVM CLASSIFICATION
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

PREDICTIONS_FILE = os.path.join(
    DATA_DIR,
    "papaya_final_svm_predictions.csv"
)

METRICS_FILE = os.path.join(
    DATA_DIR,
    "papaya_final_svm_metrics.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_DIR,
    "papaya_final_svm_confusion_matrix.png"
)

# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PAPAYA HSI FINAL SVM CLASSIFICATION")
print("=" * 70)

# ================================================================
# LOAD DATA
# ================================================================

print("\n" + "=" * 70)
print("LOADING PAPAYA 90-PC DATA")
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
# PCA FEATURES
# ================================================================

pca_features = [
    col for col in train_df.columns
    if col.startswith("PC")
]

pca_features = sorted(
    pca_features,
    key=lambda x: int(x.replace("PC", ""))
)

print("\n" + "=" * 70)
print("PCA FEATURES")
print("=" * 70)

print(
    "\nNumber of PCA components:",
    len(pca_features)
)

print("\nPCA features:")
print(pca_features)

if len(pca_features) != 90:
    raise ValueError(
        f"Expected 90 PCA components, "
        f"found {len(pca_features)}."
    )

# ================================================================
# COLUMN VALIDATION
# ================================================================

required_columns = [
    "label",
    "Sample_ID"
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

groups_train = train_df[
    "Sample_ID"
].to_numpy()

groups_test = test_df[
    "Sample_ID"
].to_numpy()

print("\n" + "=" * 70)
print("DATA MATRICES")
print("=" * 70)

print(
    "\nTraining matrix :",
    X_train.shape
)

print(
    "Testing matrix  :",
    X_test.shape
)

print(
    "Training labels :",
    y_train.shape
)

print(
    "Testing labels  :",
    y_test.shape
)

# ================================================================
# NUMERICAL VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)

print(
    "\nTraining NaN values :",
    np.isnan(X_train).sum()
)

print(
    "Testing NaN values  :",
    np.isnan(X_test).sum()
)

print(
    "Training Inf values :",
    np.isinf(X_train).sum()
)

print(
    "Testing Inf values  :",
    np.isinf(X_test).sum()
)

if not np.isfinite(X_train).all():
    raise ValueError(
        "Training data contains NaN or Inf."
    )

if not np.isfinite(X_test).all():
    raise ValueError(
        "Testing data contains NaN or Inf."
    )

# ================================================================
# SAMPLE_ID LEAKAGE CHECK
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

print(
    "\nTraining Sample_IDs:",
    len(train_sources)
)

print(
    "Testing Sample_IDs :",
    len(test_sources)
)

print(
    "Overlapping IDs    :",
    len(overlap)
)

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
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nTesting:")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)

# ================================================================
# STANDARDIZATION
# ================================================================
#
# IMPORTANT:
# StandardScaler is fitted ONLY on the training data.
#
# The test data is transformed using the parameters learned
# from the training data.
#
# ================================================================

print("\n" + "=" * 70)
print("TRAINING-DERIVED STANDARDIZATION")
print("=" * 70)

print(
    "\nFitting StandardScaler using training data only..."
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

print(
    "Transforming testing data using "
    "training-derived parameters..."
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "Training standardization completed."
)

# ================================================================
# SVM CONFIGURATION
# ================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL SVM")
print("=" * 70)

KERNEL = "rbf"
C_VALUE = 1.0
GAMMA = "scale"

print("\nKernel :", KERNEL)
print("C      :", C_VALUE)
print("Gamma  :", GAMMA)

print(
    "Training samples:",
    len(X_train_scaled)
)

print(
    "Input features:",
    X_train_scaled.shape[1]
)

# ================================================================
# TRAIN SVM
# ================================================================

model = SVC(
    kernel=KERNEL,
    C=C_VALUE,
    gamma=GAMMA
)

print("\nFitting SVM...")

model.fit(
    X_train_scaled,
    y_train
)

print(
    "SVM training completed."
)

# ================================================================
# PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test_scaled
)

print(
    "\nPredictions generated:",
    len(y_pred)
)

# ================================================================
# METRICS
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
print("SVM RESULTS")
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

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ================================================================
# PER-CLASS PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

classes = sorted(
    np.unique(y_test)
)

class_rows = []

for cls in classes:

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
    labels=classes
)

print(cm)

plt.figure(
    figsize=(9, 8)
)

plt.imshow(cm)

plt.title(
    "Papaya HSI SVM Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    range(len(classes)),
    classes
)

plt.yticks(
    range(len(classes)),
    classes
)

for i in range(
    len(classes)
):

    for j in range(
        len(classes)
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
        "SVM",

    "kernel":
        KERNEL,

    "C":
        C_VALUE,

    "gamma":
        GAMMA,

    "pca_components":
        len(pca_features),

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
print("FINAL PAPAYA HSI SVM COMPLETED")
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