"""
test_tomato_pca_sensitivity.py

PCA component sensitivity experiment for Tomato HSI.

Purpose:
    Determine whether the low baseline classification accuracy
    is caused by retaining only 2 PCA components.

Experiments:
    2, 10, 20, 30, 40, 50 PCA components

IMPORTANT:
    PCA is fitted ONLY on training samples.

    The same already-established source-cube-level train/test
    split is used for every experiment.

Classes:
    Healthy
    Alternaria alternata
    Alternaria solani
    Botrytis cinerea
    Fusarium oxysporum
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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
    "pca_sensitivity"
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

PCA_COMPONENTS_TO_TEST = [
    2,
    10,
    20,
    30,
    40,
    50
]

TARGET_VARIANCE = 0.95

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
        "LOADING TOMATO PCA SENSITIVITY DATA"
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
# GET ORIGINAL PCA FEATURES
# ============================================================

def get_pca_columns(
    train_df,
    test_df
):

    columns = [
        column
        for column in train_df.columns
        if column.startswith("PC")
    ]

    if not columns:

        raise ValueError(
            "No PCA columns found."
        )

    for column in columns:

        if column not in test_df.columns:

            raise ValueError(
                f"Missing test feature: {column}"
            )

    return columns


# ============================================================
# SOURCE CUBE LEAKAGE CHECK
# ============================================================

def check_source_leakage(
    train_df,
    test_df
):

    train_files = set(
        train_df["source_file"]
    )

    test_files = set(
        test_df["source_file"]
    )

    overlap = (
        train_files
        .intersection(test_files)
    )

    print()
    print(
        "SOURCE-CUBE LEAKAGE CHECK"
    )

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
            "Source-cube leakage detected."
        )

    print(
        "Leakage check: PASS"
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    train_df,
    test_df,
    feature_columns
):

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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if np.isnan(X_train).any():

        raise ValueError(
            "NaN values found in training data."
        )

    if np.isnan(X_test).any():

        raise ValueError(
            "NaN values found in testing data."
        )

    if np.isinf(X_train).any():

        raise ValueError(
            "Infinite values found in training data."
        )

    if np.isinf(X_test).any():

        raise ValueError(
            "Infinite values found in testing data."
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# RUN ONE PCA EXPERIMENT
# ============================================================

def run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    n_components
):

    print()
    print("-" * 75)

    print(
        f"TESTING {n_components} PCA COMPONENTS"
    )

    print("-" * 75)

    # --------------------------------------------------------
    # PCA
    #
    # IMPORTANT:
    # Fit ONLY on training data.
    # --------------------------------------------------------

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(
        X_train
    )

    X_test_pca = pca.transform(
        X_test
    )

    # --------------------------------------------------------
    # Explained variance
    # --------------------------------------------------------

    retained_variance = (
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    print(
        f"Training PCA shape : "
        f"{X_train_pca.shape}"
    )

    print(
        f"Testing PCA shape  : "
        f"{X_test_pca.shape}"
    )

    print(
        f"Variance retained  : "
        f"{retained_variance * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    print(
        "Training Random Forest..."
    )

    model.fit(
        X_train_pca,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_pca
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=CLASS_ORDER,
            zero_division=0
        )
    )

    macro_precision = np.mean(
        precision
    )

    macro_recall = np.mean(
        recall
    )

    macro_f1 = np.mean(
        f1
    )

    print(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro precision    : "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro recall       : "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1 * 100:.2f}%"
    )

    return {
        "PCA components": n_components,
        "Variance retained (%)":
            retained_variance * 100,
        "Accuracy (%)":
            accuracy * 100,
        "Macro Precision (%)":
            macro_precision * 100,
        "Macro Recall (%)":
            macro_recall * 100,
        "Macro F1 (%)":
            macro_f1 * 100
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    dataframe = pd.DataFrame(
        results
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "tomato_pca_sensitivity_results.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved results:"
    )

    print(
        output_path
    )

    return dataframe


# ============================================================
# ACCURACY GRAPH
# ============================================================

def save_accuracy_graph(
    dataframe
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        dataframe["PCA components"],
        dataframe["Accuracy (%)"],
        marker="o"
    )

    # Baseline
    plt.axhline(
        28.41,
        linestyle="--",
        label="Previous 2-PC baseline: 28.41%"
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Test Accuracy (%)"
    )

    plt.title(
        "Tomato HSI - PCA Component Sensitivity"
    )

    plt.xticks(
        PCA_COMPONENTS_TO_TEST
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        GRAPH_DIR,
        "tomato_pca_component_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved accuracy graph:"
    )

    print(
        output_path
    )


# ============================================================
# VARIANCE GRAPH
# ============================================================

def save_variance_graph(
    dataframe
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        dataframe["PCA components"],
        dataframe["Variance retained (%)"],
        marker="o"
    )

    plt.axhline(
        TARGET_VARIANCE * 100,
        linestyle="--",
        label="95% variance target"
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Explained Variance (%)"
    )

    plt.title(
        "Tomato HSI - PCA Variance vs Components"
    )

    plt.xticks(
        PCA_COMPONENTS_TO_TEST
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        GRAPH_DIR,
        "tomato_pca_component_variance.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved variance graph:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "TOMATO HSI PCA COMPONENT SENSITIVITY EXPERIMENT"
    )

    print(
        "Purpose:"
    )

    print(
        "Determine whether retaining only 2 PCA components "
        "is limiting fungal classification performance."
    )

    print()
    print(
        "Components to test:"
    )

    print(
        PCA_COMPONENTS_TO_TEST
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "PCA is fitted ONLY on training data."
    )

    print(
        "The same source-cube-level test set is used "
        "for every experiment."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_df, test_df = load_data()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = get_pca_columns(
        train_df,
        test_df
    )

    print()
    print(
        f"Available PCA features: "
        f"{len(feature_columns)}"
    )

    print(
        feature_columns
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    check_source_leakage(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(
        train_df,
        test_df,
        feature_columns
    )

    print()
    print(
        f"Training samples: "
        f"{X_train.shape[0]}"
    )

    print(
        f"Testing samples : "
        f"{X_test.shape[0]}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Our current CSV contains only 2 PCs.
    #
    # Therefore we cannot test 10/20/30/40/50 from the
    # existing PCA CSV.
    #
    # We must return to the original preprocessed tomato
    # spectral data and calculate PCA up to 50 components.
    # --------------------------------------------------------

    if len(feature_columns) < 50:

        raise ValueError(
            "\n"
            "The current tomato_pca_features.csv contains only "
            f"{len(feature_columns)} PCA components.\n\n"
            "This sensitivity experiment requires PCA to be "
            "recomputed from the original preprocessed tomato "
            "spectral data with up to 50 components.\n\n"
            "Do NOT simply duplicate PC1/PC2. "
            "The additional PCs must be calculated from the "
            "original 550-band spectral data."
        )

    # --------------------------------------------------------
    # Experiments
    # --------------------------------------------------------

    results = []

    for n_components in PCA_COMPONENTS_TO_TEST:

        result = run_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            n_components
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dataframe = save_results(
        results
    )

    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    save_accuracy_graph(
        dataframe
    )

    save_variance_graph(
        dataframe
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    header(
        "PCA SENSITIVITY SUMMARY"
    )

    print(
        dataframe.to_string(
            index=False
        )
    )

    best_row = dataframe.loc[
        dataframe["Accuracy (%)"].idxmax()
    ]

    print()
    print(
        "BEST ACCURACY"
    )

    print(
        f"PCA components : "
        f"{int(best_row['PCA components'])}"
    )

    print(
        f"Accuracy        : "
        f"{best_row['Accuracy (%)']:.2f}%"
    )

    print(
        f"Variance retained: "
        f"{best_row['Variance retained (%)']:.2f}%"
    )

    header(
        "PCA SENSITIVITY EXPERIMENT COMPLETED"
    )


if __name__ == "__main__":
    main()