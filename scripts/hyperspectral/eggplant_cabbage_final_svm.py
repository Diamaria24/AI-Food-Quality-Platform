# ================================================================
# EGGPLANT + CABBAGE FINAL SVM CLASSIFICATION
# ================================================================
#
# FINAL PCA:
#   Eggplant = 30 PCs
#   Cabbage  = 30 PCs
#
# SVM:
#   Kernel = RBF
#   C      = 1.0
#   Gamma  = scale
#
# IMPORTANT:
#   FULL training datasets are used.
#
#   Eggplant:
#       30,000 training samples
#       30,000 testing samples
#
#   Cabbage:
#       30,000 training samples
#       30,000 testing samples
#
# Standardization is fitted on training data only.
#
# ================================================================

import os
import gc
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral"
)

GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs"
)


# ================================================================
# DATASET CONFIGURATION
# ================================================================

DATASETS = {

    "Eggplant": {

        "train": os.path.join(
            PROCESSED_DIR,
            "eggplant_classification",
            "pca_30pc",
            "eggplant_train_30pc.csv"
        ),

        "test": os.path.join(
            PROCESSED_DIR,
            "eggplant_classification",
            "pca_30pc",
            "eggplant_test_30pc.csv"
        ),

        "output": os.path.join(
            PROCESSED_DIR,
            "eggplant_classification",
            "pca_30pc"
        ),

        "graphs": os.path.join(
            GRAPH_DIR,
            "eggplant_classification"
        )
    },

    "Cabbage": {

        "train": os.path.join(
            PROCESSED_DIR,
            "cabbage_classification",
            "pca_30pc",
            "cabbage_train_30pc.csv"
        ),

        "test": os.path.join(
            PROCESSED_DIR,
            "cabbage_classification",
            "pca_30pc",
            "cabbage_test_30pc.csv"
        ),

        "output": os.path.join(
            PROCESSED_DIR,
            "cabbage_classification",
            "pca_30pc"
        ),

        "graphs": os.path.join(
            GRAPH_DIR,
            "cabbage_classification"
        )
    }
}


# ================================================================
# SETTINGS
# ================================================================

N_PCA_COMPONENTS = 30

SVM_KERNEL = "rbf"

SVM_C = 1.0

SVM_GAMMA = "scale"

RANDOM_STATE = 42

CLASS_LABELS = [1, 2, 3]

CLASS_NAMES = [
    "Low",
    "Medium",
    "High"
]


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("EGGPLANT + CABBAGE FINAL SVM CLASSIFICATION")
print("=" * 70)

print("""
FINAL PCA COMPONENTS

Eggplant : 30
Cabbage  : 30

SVM

Kernel : RBF
C      : 1.0
Gamma  : scale

FULL TRAINING DATA WILL BE USED.

Eggplant:
    Training = 30,000
    Testing  = 30,000

Cabbage:
    Training = 30,000
    Testing  = 30,000

StandardScaler is fitted using training data only.
""")


# ================================================================
# PROCESS ONE DATASET
# ================================================================

def process_dataset(
    dataset_name,
    config
):

    print("\n\n")
    print("=" * 70)
    print(
        f"PROCESSING {dataset_name.upper()}"
    )
    print("=" * 70)

    train_file = config["train"]
    test_file = config["test"]

    output_dir = config["output"]
    graph_dir = config["graphs"]

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    print(
        "\nTraining file:"
    )

    print(
        train_file
    )

    print(
        "\nTesting file:"
    )

    print(
        test_file
    )

    if not os.path.exists(
        train_file
    ):

        raise FileNotFoundError(
            f"Training file not found:\n{train_file}"
        )

    if not os.path.exists(
        test_file
    ):

        raise FileNotFoundError(
            f"Testing file not found:\n{test_file}"
        )

    train_df = pd.read_csv(
        train_file
    )

    test_df = pd.read_csv(
        test_file
    )

    print(
        "\nTraining shape:",
        train_df.shape
    )

    print(
        "Testing shape :",
        test_df.shape
    )

    # ============================================================
    # PCA COLUMNS
    # ============================================================

    pca_columns = [
        f"PC{i}"
        for i in range(
            1,
            N_PCA_COMPONENTS + 1
        )
    ]

    print("\n" + "-" * 70)
    print("PCA FEATURES")
    print("-" * 70)

    print(
        pca_columns
    )

    missing_train = [
        col
        for col in pca_columns
        if col not in train_df.columns
    ]

    missing_test = [
        col
        for col in pca_columns
        if col not in test_df.columns
    ]

    required_columns = [
        "label",
        "class_name",
        "row",
        "column"
    ]

    missing_metadata_train = [
        col
        for col in required_columns
        if col not in train_df.columns
    ]

    missing_metadata_test = [
        col
        for col in required_columns
        if col not in test_df.columns
    ]

    if missing_train:

        raise ValueError(
            f"Missing PCA columns in training: "
            f"{missing_train}"
        )

    if missing_test:

        raise ValueError(
            f"Missing PCA columns in testing: "
            f"{missing_test}"
        )

    if missing_metadata_train:

        raise ValueError(
            f"Missing metadata in training: "
            f"{missing_metadata_train}"
        )

    if missing_metadata_test:

        raise ValueError(
            f"Missing metadata in testing: "
            f"{missing_metadata_test}"
        )

    print(
        "\nAll required columns found."
    )

    # ============================================================
    # CREATE MATRICES
    # ============================================================

    X_train = train_df[
        pca_columns
    ].to_numpy(
        dtype=np.float32
    )

    X_test = test_df[
        pca_columns
    ].to_numpy(
        dtype=np.float32
    )

    y_train = train_df[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    y_test = test_df[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    # ============================================================
    # DATA VALIDATION
    # ============================================================

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    print(
        "\nTraining matrix:",
        X_train.shape
    )

    print(
        "Testing matrix :",
        X_test.shape
    )

    print(
        "Training labels:",
        y_train.shape
    )

    print(
        "Testing labels :",
        y_test.shape
    )

    train_nan = int(
        np.isnan(X_train).sum()
    )

    test_nan = int(
        np.isnan(X_test).sum()
    )

    train_inf = int(
        np.isinf(X_train).sum()
    )

    test_inf = int(
        np.isinf(X_test).sum()
    )

    print(
        "\nTraining NaN:",
        train_nan
    )

    print(
        "Testing NaN :",
        test_nan
    )

    print(
        "Training Inf:",
        train_inf
    )

    print(
        "Testing Inf :",
        test_inf
    )

    if (
        train_nan > 0
        or test_nan > 0
        or train_inf > 0
        or test_inf > 0
    ):

        raise ValueError(
            "NaN or Inf values detected."
        )

    # ============================================================
    # SPATIAL LEAKAGE CHECK
    # ============================================================

    print("\n" + "=" * 70)
    print("SPATIAL LEAKAGE CHECK")
    print("=" * 70)

    train_coordinates = set(
        zip(
            train_df["row"].tolist(),
            train_df["column"].tolist()
        )
    )

    test_coordinates = set(
        zip(
            test_df["row"].tolist(),
            test_df["column"].tolist()
        )
    )

    overlap = (
        train_coordinates
        &
        test_coordinates
    )

    print(
        "\nTraining coordinates:",
        len(train_coordinates)
    )

    print(
        "Testing coordinates :",
        len(test_coordinates)
    )

    print(
        "Overlapping coordinates:",
        len(overlap)
    )

    if len(overlap) != 0:

        raise ValueError(
            "SPATIAL LEAKAGE DETECTED."
        )

    print(
        "\nSpatial leakage check: PASS"
    )

    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        train_df[
            "class_name"
        ].value_counts()
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("TESTING CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        test_df[
            "class_name"
        ].value_counts()
        .sort_index()
    )

    # ============================================================
    # STANDARDIZATION
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING-DERIVED STANDARDIZATION")
    print("=" * 70)

    print(
        "\nFitting StandardScaler using "
        "FULL training dataset only..."
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    ).astype(
        np.float32
    )

    print(
        "Training standardization completed."
    )

    print(
        "Transforming testing dataset "
        "using training-derived parameters..."
    )

    X_test_scaled = scaler.transform(
        X_test
    ).astype(
        np.float32
    )

    print(
        "Testing transformation completed."
    )

    # ============================================================
    # DELETE UNNECESSARY ARRAYS
    # ============================================================

    del X_train
    del X_test

    gc.collect()

    # ============================================================
    # TRAIN SVM
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINAL SVM")
    print("=" * 70)

    print(
        "\nKernel:",
        SVM_KERNEL
    )

    print(
        "C:",
        SVM_C
    )

    print(
        "Gamma:",
        SVM_GAMMA
    )

    print(
        "Training samples:",
        X_train_scaled.shape[0]
    )

    print(
        "Testing samples:",
        X_test_scaled.shape[0]
    )

    print(
        "Input features:",
        X_train_scaled.shape[1]
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "RBF SVM training can take a long time "
        "with 30,000 samples."
    )

    print(
        "The program may remain at "
        "'Fitting SVM...' for a while."
    )

    print(
        "\nStarting SVM training..."
    )

    start_time = time.time()

    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        random_state=RANDOM_STATE
    )

    svm.fit(
        X_train_scaled,
        y_train
    )

    training_time = (
        time.time()
        -
        start_time
    )

    print(
        "\nSVM training completed."
    )

    print(
        "Training time:",
        f"{training_time / 60:.2f} minutes"
    )

    # ============================================================
    # PREDICTIONS
    # ============================================================

    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    prediction_start = time.time()

    predictions = svm.predict(
        X_test_scaled
    )

    prediction_time = (
        time.time()
        -
        prediction_start
    )

    print(
        "Predictions generated:",
        len(predictions)
    )

    print(
        "Prediction time:",
        f"{prediction_time / 60:.2f} minutes"
    )

    # ============================================================
    # METRICS
    # ============================================================

    accuracy = (
        accuracy_score(
            y_test,
            predictions
        )
        * 100
    )

    macro_precision = (
        precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
        * 100
    )

    macro_recall = (
        recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
        * 100
    )

    macro_f1 = (
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
        * 100
    )

    # ============================================================
    # RESULTS
    # ============================================================

    print("\n" + "=" * 70)
    print(
        f"FINAL {dataset_name.upper()} SVM RESULTS"
    )
    print("=" * 70)

    print(
        f"\nTEST ACCURACY : {accuracy:.2f}%"
    )

    print(
        f"MACRO PRECISION: {macro_precision:.2f}%"
    )

    print(
        f"MACRO RECALL   : {macro_recall:.2f}%"
    )

    print(
        f"MACRO F1       : {macro_f1:.2f}%"
    )

    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(
        report
    )

    # ============================================================
    # PER CLASS PERFORMANCE
    # ============================================================

    report_dict = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    per_class_rows = []

    for class_name in CLASS_NAMES:

        per_class_rows.append({

            "class":
                class_name,

            "precision":
                report_dict[
                    class_name
                ]["precision"],

            "recall":
                report_dict[
                    class_name
                ]["recall"],

            "f1_score":
                report_dict[
                    class_name
                ]["f1-score"],

            "support":
                report_dict[
                    class_name
                ]["support"]
        })

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    print("\n" + "=" * 70)
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

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_LABELS
    )

    print(
        cm
    )

    # ============================================================
    # SAVE CONFUSION MATRIX
    # ============================================================

    cm_file = os.path.join(
        graph_dir,
        f"{dataset_name.lower()}_final_svm_confusion_matrix.csv"
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        cm_file
    )

    print(
        "\nSaved confusion matrix:"
    )

    print(
        cm_file
    )

    # ============================================================
    # SAVE PREDICTIONS
    # ============================================================

    prediction_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_final_svm_predictions.csv"
    )

    predictions_df = test_df[
        [
            "row",
            "column",
            "label",
            "class_name"
        ]
    ].copy()

    predictions_df[
        "predicted_label"
    ] = predictions

    predictions_df[
        "predicted_class"
    ] = [
        CLASS_NAMES[
            int(pred - 1)
        ]
        for pred in predictions
    ]

    predictions_df[
        "correct"
    ] = (
        predictions_df["label"]
        ==
        predictions_df["predicted_label"]
    )

    predictions_df.to_csv(
        prediction_file,
        index=False
    )

    print(
        "\nSaved predictions:"
    )

    print(
        prediction_file
    )

    # ============================================================
    # SAVE METRICS
    # ============================================================

    metrics_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_final_svm_metrics.csv"
    )

    metrics_df = pd.DataFrame([{

        "dataset":
            dataset_name,

        "model":
            "SVM",

        "kernel":
            SVM_KERNEL,

        "C":
            SVM_C,

        "gamma":
            SVM_GAMMA,

        "pca_components":
            N_PCA_COMPONENTS,

        "training_samples":
            len(y_train),

        "testing_samples":
            len(y_test),

        "accuracy_percent":
            accuracy,

        "macro_precision_percent":
            macro_precision,

        "macro_recall_percent":
            macro_recall,

        "macro_f1_percent":
            macro_f1,

        "training_time_minutes":
            training_time / 60,

        "prediction_time_minutes":
            prediction_time / 60,

        "spatial_coordinate_overlap":
            len(overlap)
    }])

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    print(
        "\nSaved metrics:"
    )

    print(
        metrics_file
    )

    # ============================================================
    # FINAL
    # ============================================================

    print("\n" + "=" * 70)
    print(
        f"FINAL {dataset_name.upper()} SVM COMPLETED"
    )
    print("=" * 70)

    print(
        f"""
Final test accuracy : {accuracy:.2f}%
Macro F1            : {macro_f1:.2f}%

PCA components      : {N_PCA_COMPONENTS}

Training samples    : {len(y_train):,}
Testing samples     : {len(y_test):,}

SVM kernel          : {SVM_KERNEL}
C                   : {SVM_C}
Gamma               : {SVM_GAMMA}

Training time       : {training_time / 60:.2f} minutes
Prediction time     : {prediction_time / 60:.2f} minutes

Spatial overlap     : {len(overlap)}

Classification outputs:
{output_dir}

Graphs:
{graph_dir}
"""
    )

    # ============================================================
    # CLEANUP
    # ============================================================

    del train_df
    del test_df
    del X_train_scaled
    del X_test_scaled
    del y_train
    del y_test
    del predictions
    del svm
    del scaler

    gc.collect()

    return {

        "dataset":
            dataset_name,

        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "training_time_minutes":
            training_time / 60,

        "prediction_time_minutes":
            prediction_time / 60
    }


# ================================================================
# RUN EGGPLANT
# ================================================================

eggplant_result = process_dataset(
    "Eggplant",
    DATASETS["Eggplant"]
)


# ================================================================
# RUN CABBAGE
# ================================================================

cabbage_result = process_dataset(
    "Cabbage",
    DATASETS["Cabbage"]
)


# ================================================================
# COMBINED SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("EGGPLANT + CABBAGE SVM SUMMARY")
print("=" * 70)

combined_df = pd.DataFrame([
    eggplant_result,
    cabbage_result
])

print(
    combined_df.to_string(
        index=False
    )
)

combined_file = os.path.join(
    PROCESSED_DIR,
    "eggplant_cabbage_final_svm_summary.csv"
)

combined_df.to_csv(
    combined_file,
    index=False
)

print(
    "\nCombined summary saved:"
)

print(
    combined_file
)


# ================================================================
# FINAL
# ================================================================

print("\n" + "=" * 70)
print("EGGPLANT + CABBAGE FINAL SVM COMPLETED")
print("=" * 70)

print("""
NEXT STEP:

Compare the final models:

EGGPLANT
    Random Forest vs SVM

CABBAGE
    Random Forest vs SVM

Comparison metrics:

    Accuracy
    Macro Precision
    Macro Recall
    Macro F1
    Per-class performance
    Confusion matrices
    Training time
""")

print("=" * 70)