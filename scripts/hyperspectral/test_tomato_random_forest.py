"""
test_tomato_random_forest.py

Random Forest classification for the real Tomato HSI dataset.

Input:
    tomato_train.csv
    tomato_test.csv

Features:
    PCA components

Classes:
    Healthy
    Alternaria alternata
    Alternaria solani
    Botrytis cinerea
    Fusarium oxysporum

IMPORTANT:
The train/test split was already performed at SOURCE-CUBE level.
Therefore, this script does NOT perform another random split.
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
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "tomato_classification"
)

TRAIN_FILE = os.path.join(
    INPUT_DIR,
    "tomato_train.csv"
)

TEST_FILE = os.path.join(
    INPUT_DIR,
    "tomato_test.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "tomato_classification"
)

GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "tomato_classification"
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

RANDOM_STATE = 42

N_ESTIMATORS = 200

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
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    header(
        "LOADING TOMATO CLASSIFICATION DATA"
    )

    print(
        f"Training file:\n{TRAIN_FILE}"
    )

    print(
        f"\nTesting file:\n{TEST_FILE}"
    )

    if not os.path.exists(TRAIN_FILE):

        raise FileNotFoundError(
            "Training file not found."
        )

    if not os.path.exists(TEST_FILE):

        raise FileNotFoundError(
            "Testing file not found."
        )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    print()
    print(
        f"Training shape : {train_df.shape}"
    )

    print(
        f"Testing shape  : {test_df.shape}"
    )

    return train_df, test_df


# ============================================================
# IDENTIFY PCA FEATURES
# ============================================================

def get_pca_columns(train_df, test_df):

    train_pca = [
        column
        for column in train_df.columns
        if column.startswith("PC")
    ]

    test_pca = [
        column
        for column in test_df.columns
        if column.startswith("PC")
    ]

    if train_pca != test_pca:

        raise ValueError(
            "Training and testing PCA columns do not match."
        )

    if not train_pca:

        raise ValueError(
            "No PCA feature columns found."
        )

    return train_pca


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
    ]

    X_test = test_df[
        feature_columns
    ]

    y_train = train_df[
        "class"
    ]

    y_test = test_df[
        "class"
    ]

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

    print()
    print(
        f"Training NaN values : "
        f"{X_train.isna().sum().sum()}"
    )

    print(
        f"Testing NaN values  : "
        f"{X_test.isna().sum().sum()}"
    )

    print(
        f"Training Inf values : "
        f"{np.isinf(X_train.to_numpy()).sum()}"
    )

    print(
        f"Testing Inf values  : "
        f"{np.isinf(X_test.to_numpy()).sum()}"
    )

    # --------------------------------------------------------
    # Source file leakage check
    # --------------------------------------------------------

    train_files = set(
        train_df["source_file"]
    )

    test_files = set(
        test_df["source_file"]
    )

    overlap = train_files.intersection(
        test_files
    )

    print()
    print(
        f"Training source cubes: "
        f"{len(train_files)}"
    )

    print(
        f"Testing source cubes : "
        f"{len(test_files)}"
    )

    print(
        f"Overlapping cubes    : "
        f"{len(overlap)}"
    )

    if overlap:

        raise ValueError(
            "DATA LEAKAGE DETECTED!"
        )

    print(
        "Source-cube leakage check: PASS"
    )

    return (
        X_train.to_numpy(dtype=np.float32),
        X_test.to_numpy(dtype=np.float32),
        y_train.to_numpy(),
        y_test.to_numpy()
    )


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train
):

    header(
        "TRAINING RANDOM FOREST"
    )

    print(
        f"Number of trees : {N_ESTIMATORS}"
    )

    print(
        f"Random state    : {RANDOM_STATE}"
    )

    print(
        f"Training samples: {X_train.shape[0]}"
    )

    print(
        f"Input features  : {X_train.shape[1]}"
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
# PREDICTION
# ============================================================

def predict(
    model,
    X_test
):

    header(
        "GENERATING TEST PREDICTIONS"
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    print(
        f"Predictions generated: "
        f"{len(predictions)}"
    )

    return predictions, probabilities


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
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

    print()
    print(
        "CLASSIFICATION REPORT"
    )

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        digits=4,
        zero_division=0
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Precision / Recall / F1
    # --------------------------------------------------------

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=CLASS_ORDER,
            zero_division=0
        )
    )

    results = pd.DataFrame({
        "class": CLASS_ORDER,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support
    })

    print()
    print(
        "PER-CLASS PERFORMANCE"
    )

    print(
        results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_output = os.path.join(
        OUTPUT_DIR,
        "tomato_random_forest_metrics.csv"
    )

    results.to_csv(
        metrics_output,
        index=False
    )

    print()
    print(
        f"Saved metrics:"
    )

    print(
        metrics_output
    )

    return accuracy, results


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

    print(
        matrix
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_ORDER
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d"
    )

    ax.set_title(
        "Tomato HSI - Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    output_path = os.path.join(
        GRAPH_DIR,
        "tomato_random_forest_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved confusion matrix:"
    )

    print(
        output_path
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    feature_columns
):

    header(
        "FEATURE IMPORTANCE"
    )

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
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

    output_csv = os.path.join(
        OUTPUT_DIR,
        "tomato_random_forest_feature_importance.csv"
    )

    importance_df.to_csv(
        output_csv,
        index=False
    )

    print()
    print(
        f"Saved feature importance:"
    )

    print(
        output_csv
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.xlabel(
        "PCA Feature"
    )

    plt.ylabel(
        "Importance"
    )

    plt.title(
        "Tomato HSI - Random Forest Feature Importance"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_graph = os.path.join(
        GRAPH_DIR,
        "tomato_random_forest_feature_importance.png"
    )

    plt.savefig(
        output_graph,
        dpi=200
    )

    plt.close()

    print(
        f"Saved feature importance graph:"
    )

    print(
        output_graph
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    test_df,
    predictions,
    probabilities
):

    output = test_df.copy()

    output["predicted_class"] = predictions

    output["correct"] = (
        output["class"]
        ==
        output["predicted_class"]
    )

    # Add confidence
    output["prediction_confidence"] = (
        probabilities.max(
            axis=1
        )
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "tomato_random_forest_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved predictions:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "TOMATO HSI RANDOM FOREST CLASSIFICATION"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_df, test_df = load_data()

    # --------------------------------------------------------
    # PCA columns
    # --------------------------------------------------------

    feature_columns = get_pca_columns(
        train_df,
        test_df
    )

    print()
    print(
        f"PCA features used: "
        f"{feature_columns}"
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

    model = train_random_forest(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions, probabilities = predict(
        model,
        X_test
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    accuracy, results = evaluate(
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
        predictions,
        probabilities
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    header(
        "TOMATO RANDOM FOREST COMPLETED"
    )

    print(
        f"Final test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print(
        "The model was evaluated on completely "
        "unseen source cubes."
    )

    print()
    print(
        "Classification outputs saved to:"
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