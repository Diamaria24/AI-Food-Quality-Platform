"""
test_tomato_rf_20pc.py

Final analysis of the Tomato HSI Random Forest model
using the selected 20 PCA components.

Configuration
-------------
Original spectral bands : 550
PCA components          : 20
Random Forest trees     : 200
Random state            : 42

Train/test separation
---------------------
Source-cube level.

Training:
    one complete MAT cube per class

Testing:
    one completely unseen MAT cube per class

Outputs
-------
1. Classification report
2. Confusion matrix
3. Per-class metrics
4. PCA feature importance
5. Predictions
6. Actual vs predicted distribution
7. CSV files
8. Graphs
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

CLASSIFICATION_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "tomato_classification"
)

SENSITIVITY_DIR = os.path.join(
    CLASSIFICATION_DIR,
    "pca_sensitivity"
)

TRAIN_FILE = os.path.join(
    CLASSIFICATION_DIR,
    "tomato_train.csv"
)

TEST_FILE = os.path.join(
    CLASSIFICATION_DIR,
    "tomato_test.csv"
)

OUTPUT_DIR = os.path.join(
    CLASSIFICATION_DIR,
    "rf_20pc"
)

GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "tomato_classification",
    "rf_20pc"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

N_COMPONENTS = 20

N_ESTIMATORS = 200

RANDOM_STATE = 42

CLASS_ORDER = [
    "Healthy",
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum"
]


# ============================================================
# HEADER
# ============================================================

def header(title):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    header(
        "LOADING TOMATO CLASSIFICATION DATA"
    )

    if not os.path.exists(TRAIN_FILE):

        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    if not os.path.exists(TEST_FILE):

        raise FileNotFoundError(
            f"Testing file not found:\n{TEST_FILE}"
        )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Training shape : {train_df.shape}"
    )

    print(
        f"Testing shape  : {test_df.shape}"
    )

    return train_df, test_df


# ============================================================
# FIND PCA COLUMNS
# ============================================================

def find_pca_columns(
    train_df,
    test_df
):

    all_train_pcs = [
        column
        for column in train_df.columns
        if column.startswith("PC")
    ]

    all_test_pcs = [
        column
        for column in test_df.columns
        if column.startswith("PC")
    ]

    if len(all_train_pcs) < N_COMPONENTS:

        raise ValueError(
            f"Only {len(all_train_pcs)} PCA components "
            f"are available in the classification CSV.\n"
            f"{N_COMPONENTS} components are required.\n\n"
            "This script must use the REAL 20-PC dataset "
            "generated from the original 550-band data."
        )

    selected_columns = [
        f"PC{i}"
        for i in range(
            1,
            N_COMPONENTS + 1
        )
    ]

    for column in selected_columns:

        if column not in train_df.columns:

            raise ValueError(
                f"Missing training column: {column}"
            )

        if column not in test_df.columns:

            raise ValueError(
                f"Missing testing column: {column}"
            )

    return selected_columns


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(
    train_df,
    test_df,
    feature_columns
):

    header(
        "DATA VALIDATION"
    )

    X_train = train_df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    X_test = test_df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    y_train = train_df[
        "class"
    ].to_numpy()

    y_test = test_df[
        "class"
    ].to_numpy()

    print(
        f"Training matrix : {X_train.shape}"
    )

    print(
        f"Testing matrix  : {X_test.shape}"
    )

    print(
        f"Training labels : {y_train.shape}"
    )

    print(
        f"Testing labels  : {y_test.shape}"
    )

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    train_nan = np.isnan(
        X_train
    ).sum()

    test_nan = np.isnan(
        X_test
    ).sum()

    train_inf = np.isinf(
        X_train
    ).sum()

    test_inf = np.isinf(
        X_test
    ).sum()

    print()
    print(
        f"Training NaN : {train_nan}"
    )

    print(
        f"Testing NaN  : {test_nan}"
    )

    print(
        f"Training Inf : {train_inf}"
    )

    print(
        f"Testing Inf  : {test_inf}"
    )

    if train_nan or test_nan:

        raise ValueError(
            "NaN values detected."
        )

    if train_inf or test_inf:

        raise ValueError(
            "Infinite values detected."
        )

    # --------------------------------------------------------
    # Class validation
    # --------------------------------------------------------

    print()
    print(
        "TRAINING CLASS DISTRIBUTION"
    )

    print(
        train_df["class"].value_counts()
    )

    print()
    print(
        "TEST CLASS DISTRIBUTION"
    )

    print(
        test_df["class"].value_counts()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# SOURCE CUBE LEAKAGE CHECK
# ============================================================

def check_leakage(
    train_df,
    test_df
):

    header(
        "SOURCE-CUBE LEAKAGE CHECK"
    )

    if (
        "source_file" not in train_df.columns
        or
        "source_file" not in test_df.columns
    ):

        raise ValueError(
            "source_file column is required."
        )

    train_sources = set(
        train_df["source_file"]
    )

    test_sources = set(
        test_df["source_file"]
    )

    overlap = (
        train_sources
        .intersection(
            test_sources
        )
    )

    print(
        f"Training source cubes: "
        f"{len(train_sources)}"
    )

    print(
        f"Testing source cubes : "
        f"{len(test_sources)}"
    )

    print(
        f"Overlapping cubes    : "
        f"{len(overlap)}"
    )

    if overlap:

        print(
            "OVERLAPPING FILES:"
        )

        for item in sorted(
            overlap
        ):

            print(
                f" - {item}"
            )

        raise ValueError(
            "Source-cube leakage detected."
        )

    print(
        "Source-cube leakage check: PASS"
    )


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(
    X_train,
    y_train
):

    header(
        "TRAINING RANDOM FOREST"
    )

    print(
        f"PCA components : {N_COMPONENTS}"
    )

    print(
        f"Number of trees: {N_ESTIMATORS}"
    )

    print(
        f"Random state   : {RANDOM_STATE}"
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Input features: "
        f"{X_train.shape[1]}"
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    print()
    print(
        "Fitting model..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest training completed."
    )

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X_test
):

    header(
        "GENERATING TEST PREDICTIONS"
    )

    predictions = model.predict(
        X_test
    )

    print(
        f"Predictions generated: "
        f"{len(predictions)}"
    )

    return predictions


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    y_test,
    predictions
):

    header(
        "RANDOM FOREST RESULTS"
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print(
        f"TEST ACCURACY: "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        zero_division=0
    )

    print()
    print(
        "CLASSIFICATION REPORT"
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Numeric metrics
    # --------------------------------------------------------

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=CLASS_ORDER,
            zero_division=0
        )
    )

    metrics_df = pd.DataFrame({

        "class":
            CLASS_ORDER,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1,

        "support":
            support
    })

    print()
    print(
        "PER-CLASS PERFORMANCE"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    macro_precision = (
        precision.mean()
    )

    macro_recall = (
        recall.mean()
    )

    macro_f1 = (
        f1.mean()
    )

    summary_df = pd.DataFrame({

        "metric": [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1"
        ],

        "value": [
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1
        ]
    })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "tomato_rf_20pc_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "tomato_rf_20pc_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        "tomato_rf_20pc_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    print()
    print(
        f"Saved per-class metrics:"
    )

    print(
        metrics_path
    )

    print(
        f"Saved summary:"
    )

    print(
        summary_path
    )

    print(
        f"Saved classification report:"
    )

    print(
        report_path
    )

    return (
        accuracy,
        metrics_df,
        summary_df
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_test,
    predictions
):

    header(
        "CONFUSION MATRIX"
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_ORDER
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASS_ORDER,
        columns=CLASS_ORDER
    )

    print(
        matrix_df
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "tomato_rf_20pc_confusion_matrix.csv"
    )

    matrix_df.to_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Tomato HSI - Random Forest Confusion Matrix\n20 PCA Components"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(CLASS_ORDER)
    )

    plt.xticks(
        tick_positions,
        CLASS_ORDER,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_positions,
        CLASS_ORDER
    )

    threshold = (
        matrix.max() / 2.0
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white"
                if matrix[i, j] > threshold
                else "black"
            )

    plt.ylabel(
        "Actual Class"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.tight_layout()

    graph_path = os.path.join(
        GRAPH_DIR,
        "tomato_rf_20pc_confusion_matrix.png"
    )

    plt.savefig(
        graph_path,
        dpi=200
    )

    plt.close()

    print()
    print(
        f"Saved confusion matrix CSV:"
    )

    print(
        csv_path
    )

    print(
        f"Saved confusion matrix graph:"
    )

    print(
        graph_path
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    feature_columns
):

    header(
        "PCA FEATURE IMPORTANCE"
    )

    importances = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "tomato_rf_20pc_feature_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    sorted_plot = (
        importance_df
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        sorted_plot["feature"],
        sorted_plot["importance"]
    )

    plt.xlabel(
        "Random Forest Importance"
    )

    plt.ylabel(
        "PCA Component"
    )

    plt.title(
        "Tomato HSI - PCA Feature Importance\n20 PCA Components"
    )

    plt.tight_layout()

    graph_path = os.path.join(
        GRAPH_DIR,
        "tomato_rf_20pc_feature_importance.png"
    )

    plt.savefig(
        graph_path,
        dpi=200
    )

    plt.close()

    print()
    print(
        f"Saved feature importance CSV:"
    )

    print(
        csv_path
    )

    print(
        f"Saved feature importance graph:"
    )

    print(
        graph_path
    )

    return importance_df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test_df,
    predictions
):

    header(
        "SAVING TEST PREDICTIONS"
    )

    prediction_df = pd.DataFrame({

        "source_file":
            test_df["source_file"].values,

        "actual_class":
            test_df["class"].values,

        "predicted_class":
            predictions
    })

    # --------------------------------------------------------
    # Correct / incorrect
    # --------------------------------------------------------

    prediction_df[
        "correct"
    ] = (
        prediction_df[
            "actual_class"
        ]
        ==
        prediction_df[
            "predicted_class"
        ]
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "tomato_rf_20pc_predictions.csv"
    )

    prediction_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved predictions:"
    )

    print(
        output_path
    )

    return prediction_df


# ============================================================
# ACTUAL VS PREDICTED DISTRIBUTION
# ============================================================

def save_prediction_distribution(
    y_test,
    predictions
):

    header(
        "ACTUAL VS PREDICTED CLASS DISTRIBUTION"
    )

    actual_counts = (
        pd.Series(
            y_test
        )
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0
        )
    )

    predicted_counts = (
        pd.Series(
            predictions
        )
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0
        )
    )

    distribution_df = pd.DataFrame({

        "class":
            CLASS_ORDER,

        "actual":
            actual_counts.values,

        "predicted":
            predicted_counts.values
    })

    print(
        distribution_df.to_string(
            index=False
        )
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "tomato_rf_20pc_class_distribution.csv"
    )

    distribution_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    x = np.arange(
        len(CLASS_ORDER)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        x - width / 2,
        actual_counts.values,
        width,
        label="Actual"
    )

    plt.bar(
        x + width / 2,
        predicted_counts.values,
        width,
        label="Predicted"
    )

    plt.xticks(
        x,
        CLASS_ORDER,
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.xlabel(
        "Class"
    )

    plt.title(
        "Tomato HSI - Actual vs Predicted Class Distribution"
    )

    plt.legend()

    plt.tight_layout()

    graph_path = os.path.join(
        GRAPH_DIR,
        "tomato_rf_20pc_class_distribution.png"
    )

    plt.savefig(
        graph_path,
        dpi=200
    )

    plt.close()

    print()
    print(
        f"Saved distribution CSV:"
    )

    print(
        csv_path
    )

    print(
        f"Saved distribution graph:"
    )

    print(
        graph_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "TOMATO HSI RANDOM FOREST - FINAL 20 PC ANALYSIS"
    )

    print(
        f"PCA components : {N_COMPONENTS}"
    )

    print(
        f"Random Forest trees: {N_ESTIMATORS}"
    )

    print(
        f"Random state: {RANDOM_STATE}"
    )

    print()
    print(
        "This experiment evaluates the selected "
        "20-PC configuration."
    )

    print(
        "The test data consists of completely unseen "
        "source cubes."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_df, test_df = (
        load_data()
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = (
        find_pca_columns(
            train_df,
            test_df
        )
    )

    print()
    print(
        "PCA features used:"
    )

    print(
        feature_columns
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    check_leakage(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = validate_data(
        train_df,
        test_df,
        feature_columns
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = (
        generate_predictions(
            model,
            X_test
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    (
        accuracy,
        metrics_df,
        summary_df
    ) = calculate_metrics(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    save_feature_importance(
        model,
        feature_columns
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    save_predictions(
        test_df,
        predictions
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    save_prediction_distribution(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    header(
        "TOMATO 20-PC RANDOM FOREST ANALYSIS COMPLETED"
    )

    print(
        f"Final test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print(
        "Configuration:"
    )

    print(
        "550 spectral bands"
    )

    print(
        "20 PCA components"
    )

    print(
        "Random Forest: 200 trees"
    )

    print(
        "Source-cube-level train/test split"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Graphs saved to:"
    )

    print(
        GRAPH_DIR
    )


if __name__ == "__main__":

    main()