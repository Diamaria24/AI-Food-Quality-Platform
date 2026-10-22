# ================================================================
# EGGPLANT + CABBAGE FINAL RANDOM FOREST CLASSIFICATION
# ================================================================
#
# Final PCA:
#   Eggplant = 30 PCs
#   Cabbage  = 30 PCs
#
# Model:
#   Random Forest
#   Trees = 200
#   Random State = 42
#
# ================================================================

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
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

N_TREES = 200
RANDOM_STATE = 42

N_PCA_COMPONENTS = 30

CLASS_NAMES = [
    "Low",
    "Medium",
    "High"
]


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("EGGPLANT + CABBAGE FINAL RANDOM FOREST CLASSIFICATION")
print("=" * 70)

print("""
FINAL PCA COMPONENTS

Eggplant : 30
Cabbage  : 30

RANDOM FOREST

Trees       : 200
Random State: 42

The model will be evaluated on spatially separated,
completely unseen testing samples.
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
    # LOAD TRAINING DATA
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
    # PCA COLUMN DETECTION
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

    required_metadata = [
        "label",
        "class_name",
        "row",
        "column"
    ]

    missing_metadata_train = [
        col
        for col in required_metadata
        if col not in train_df.columns
    ]

    missing_metadata_test = [
        col
        for col in required_metadata
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
    # DATA MATRICES
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
    # TRAIN RANDOM FOREST
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINAL RANDOM FOREST")
    print("=" * 70)

    print(
        "\nNumber of trees:",
        N_TREES
    )

    print(
        "Random state:",
        RANDOM_STATE
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    print(
        "Input features:",
        X_train.shape[1]
    )

    print(
        "\nFitting model..."
    )

    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest training completed."
    )

    # ============================================================
    # PREDICTIONS
    # ============================================================

    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    predictions = rf.predict(
        X_test
    )

    print(
        "Predictions generated:",
        len(predictions)
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
        f"FINAL {dataset_name.upper()} RANDOM FOREST RESULTS"
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
        labels=[1, 2, 3],
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
        labels=[1, 2, 3],
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
        labels=[1, 2, 3]
    )

    print(
        cm
    )

    cm_file = os.path.join(
        graph_dir,
        f"{dataset_name.lower()}_final_random_forest_confusion_matrix.png"
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"{dataset_name} "
        "Final Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        cm_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved confusion matrix:"
    )

    print(
        cm_file
    )

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    print("\n" + "=" * 70)
    print("PCA FEATURE IMPORTANCE")
    print("=" * 70)

    feature_importance_df = pd.DataFrame({

        "feature":
            pca_columns,

        "importance":
            rf.feature_importances_
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        feature_importance_df.to_string(
            index=False
        )
    )

    importance_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_final_random_forest_feature_importance.csv"
    )

    feature_importance_df.to_csv(
        importance_file,
        index=False
    )

    print(
        "\nSaved feature importance:"
    )

    print(
        importance_file
    )

    # ============================================================
    # FEATURE IMPORTANCE GRAPH
    # ============================================================

    importance_graph = os.path.join(
        graph_dir,
        f"{dataset_name.lower()}_final_random_forest_feature_importance.png"
    )

    plot_df = feature_importance_df.sort_values(
        "importance",
        ascending=True
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["importance"]
    )

    plt.xlabel(
        "Random Forest Importance"
    )

    plt.ylabel(
        "PCA Component"
    )

    plt.title(
        f"{dataset_name} "
        "Random Forest PCA Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        importance_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved feature importance graph:"
    )

    print(
        importance_graph
    )

    # ============================================================
    # SAVE PREDICTIONS
    # ============================================================

    prediction_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_final_random_forest_predictions.csv"
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
        f"{dataset_name.lower()}_final_random_forest_metrics.csv"
    )

    metrics_df = pd.DataFrame([{

        "dataset":
            dataset_name,

        "model":
            "Random Forest",

        "pca_components":
            N_PCA_COMPONENTS,

        "variance_retained_percent":
            np.nan,

        "n_trees":
            N_TREES,

        "random_state":
            RANDOM_STATE,

        "training_samples":
            len(X_train),

        "testing_samples":
            len(X_test),

        "accuracy_percent":
            accuracy,

        "macro_precision_percent":
            macro_precision,

        "macro_recall_percent":
            macro_recall,

        "macro_f1_percent":
            macro_f1,

        "spatial_coordinate_overlap":
            len(overlap)
    }])

    # Try to retrieve variance from the existing PCA summary
    summary_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_pca_30pc_summary.csv"
    )

    if os.path.exists(
        summary_file
    ):

        try:

            previous_summary = pd.read_csv(
                summary_file
            )

            if (
                "variance_retained_percent"
                in previous_summary.columns
            ):

                metrics_df[
                    "variance_retained_percent"
                ] = previous_summary.loc[
                    0,
                    "variance_retained_percent"
                ]

        except Exception:

            pass

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
        f"FINAL {dataset_name.upper()} RANDOM FOREST COMPLETED"
    )
    print("=" * 70)

    print(
        f"""
Final test accuracy : {accuracy:.2f}%
Macro F1            : {macro_f1:.2f}%

PCA components      : {N_PCA_COMPONENTS}

Training samples    : {len(X_train):,}
Testing samples     : {len(X_test):,}

Spatial overlap     : {len(overlap)}

The model was evaluated on spatially
unseen testing samples.

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
    del X_train
    del X_test
    del y_train
    del y_test
    del predictions
    del rf

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
            macro_f1
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
# COMBINED RESULTS
# ================================================================

print("\n" + "=" * 70)
print("EGGPLANT + CABBAGE RANDOM FOREST SUMMARY")
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
    "eggplant_cabbage_final_random_forest_summary.csv"
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
print("EGGPLANT + CABBAGE FINAL RANDOM FOREST COMPLETED")
print("=" * 70)

print("""
NEXT STEP:

Use the SAME 30-PC datasets for SVM.

Then compare:

        Random Forest
              vs
             SVM

using:

    Accuracy
    Macro Precision
    Macro Recall
    Macro F1
    Per-class performance
    Confusion matrices
""")

print("=" * 70)