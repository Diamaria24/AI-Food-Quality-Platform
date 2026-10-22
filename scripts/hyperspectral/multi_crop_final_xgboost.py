# ================================================================
# MULTI-CROP FINAL XGBOOST CLASSIFICATION
# ================================================================
#
# THIRD CLASSIFIER COMPARISON
#
# Models:
#   1. Random Forest
#   2. SVM
#   3. XGBoost
#
# Crops:
#   Tomato
#   Papaya
#   Eggplant
#   Cabbage
#
# FINAL PCA:
#   Tomato   = 20 PCs
#   Papaya   = 90 PCs
#   Eggplant = 30 PCs
#   Cabbage  = 30 PCs
#
# IMPORTANT:
#   Existing train/test datasets are used.
#   No new split is created.
#
# The script supports both:
#   - numeric labels
#   - text labels
#
# ================================================================


import os
import gc
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

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

    "Tomato": {

        "train": os.path.join(
            PROCESSED_DIR,
            "tomato_classification",
            "pca_20pc",
            "tomato_train_20pc.csv"
        ),

        "test": os.path.join(
            PROCESSED_DIR,
            "tomato_classification",
            "pca_20pc",
            "tomato_test_20pc.csv"
        ),

        "output": os.path.join(
            PROCESSED_DIR,
            "tomato_classification",
            "pca_20pc"
        ),

        "graphs": os.path.join(
            GRAPH_DIR,
            "tomato_classification"
        ),

        "pca_components": 20
    },


    "Papaya": {

        "train": os.path.join(
            PROCESSED_DIR,
            "papaya_classification",
            "pca_90pc",
            "papaya_train_90pc.csv"
        ),

        "test": os.path.join(
            PROCESSED_DIR,
            "papaya_classification",
            "pca_90pc",
            "papaya_test_90pc.csv"
        ),

        "output": os.path.join(
            PROCESSED_DIR,
            "papaya_classification",
            "pca_90pc"
        ),

        "graphs": os.path.join(
            GRAPH_DIR,
            "papaya_classification"
        ),

        "pca_components": 90
    },


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
        ),

        "pca_components": 30
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
        ),

        "pca_components": 30
    }
}


# ================================================================
# XGBOOST PARAMETERS
# ================================================================

N_ESTIMATORS = 300

MAX_DEPTH = 6

LEARNING_RATE = 0.05

SUBSAMPLE = 0.8

COLSAMPLE_BYTREE = 0.8

RANDOM_STATE = 42

N_JOBS = -1


# ================================================================
# POSSIBLE LABEL COLUMN NAMES
# ================================================================

LABEL_CANDIDATES = [

    "label",
    "class",
    "Label",
    "Class",
    "Label_6Class",
    "label_6class",
    "Label_3Class",
    "label_3class"
]


# ================================================================
# FIND LABEL COLUMN
# ================================================================

def find_label_column(df):

    # ------------------------------------------------------------
    # Exact known names
    # ------------------------------------------------------------

    for candidate in LABEL_CANDIDATES:

        if candidate in df.columns:

            return candidate

    # ------------------------------------------------------------
    # Case-insensitive matching
    # ------------------------------------------------------------

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in LABEL_CANDIDATES:

        key = candidate.lower()

        if key in lower_map:

            return lower_map[key]

    # ------------------------------------------------------------
    # Fallback detection
    # ------------------------------------------------------------

    possible = []

    for col in df.columns:

        if str(col).lower().startswith("pc"):

            continue

        try:

            unique_count = df[col].nunique()

            if 2 <= unique_count <= 20:

                possible.append(
                    (col, unique_count)
                )

        except Exception:

            pass

    if len(possible) == 1:

        return possible[0][0]

    if len(possible) > 1:

        # Prefer likely label names
        for col, _ in possible:

            name = str(col).lower()

            if (
                "label" in name
                or
                "class" in name
            ):

                return col

        # Otherwise smallest cardinality
        possible.sort(
            key=lambda x: x[1]
        )

        return possible[0][0]

    raise ValueError(
        "\nCould not identify label column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ================================================================
# LEAKAGE CHECK
# ================================================================

def perform_leakage_checks(
    train_df,
    test_df
):

    print("\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)

    # ------------------------------------------------------------
    # SOURCE FILE CHECK
    # ------------------------------------------------------------

    source_overlap_count = np.nan

    if (
        "source_file" in train_df.columns
        and
        "source_file" in test_df.columns
    ):

        train_sources = set(
            train_df["source_file"]
        )

        test_sources = set(
            test_df["source_file"]
        )

        source_overlap = (
            train_sources
            &
            test_sources
        )

        source_overlap_count = len(
            source_overlap
        )

        print(
            "\nTraining source files:",
            len(train_sources)
        )

        print(
            "Testing source files :",
            len(test_sources)
        )

        print(
            "Overlapping source files:",
            source_overlap_count
        )

        if source_overlap_count != 0:

            raise ValueError(
                "SOURCE-FILE LEAKAGE DETECTED."
            )

        print(
            "Source-file leakage: PASS"
        )

    else:

        print(
            "\nsource_file column not available."
        )

        print(
            "Source-file leakage: NOT APPLICABLE"
        )

    # ------------------------------------------------------------
    # SAMPLE ID CHECK
    # ------------------------------------------------------------

    sample_id_overlap_count = np.nan

    if (
        "Sample_ID" in train_df.columns
        and
        "Sample_ID" in test_df.columns
    ):

        train_ids = set(
            train_df["Sample_ID"]
        )

        test_ids = set(
            test_df["Sample_ID"]
        )

        sample_overlap = (
            train_ids
            &
            test_ids
        )

        sample_id_overlap_count = len(
            sample_overlap
        )

        print(
            "\nTraining Sample_IDs:",
            len(train_ids)
        )

        print(
            "Testing Sample_IDs :",
            len(test_ids)
        )

        print(
            "Overlapping Sample_IDs:",
            sample_id_overlap_count
        )

        if sample_id_overlap_count != 0:

            raise ValueError(
                "SAMPLE_ID LEAKAGE DETECTED."
            )

        print(
            "Sample_ID leakage: PASS"
        )

    else:

        print(
            "\nSample_ID column not available."
        )

        print(
            "Sample_ID leakage: NOT APPLICABLE"
        )

    # ------------------------------------------------------------
    # COORDINATE CHECK
    # ------------------------------------------------------------

    coordinate_overlap_count = np.nan

    if (
        "row" in train_df.columns
        and
        "column" in train_df.columns
        and
        "row" in test_df.columns
        and
        "column" in test_df.columns
    ):

        train_coordinates = set(
            zip(
                train_df["row"],
                train_df["column"]
            )
        )

        test_coordinates = set(
            zip(
                test_df["row"],
                test_df["column"]
            )
        )

        coordinate_overlap = (
            train_coordinates
            &
            test_coordinates
        )

        coordinate_overlap_count = len(
            coordinate_overlap
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
            "Coordinate overlap:",
            coordinate_overlap_count
        )

        if coordinate_overlap_count != 0:

            raise ValueError(
                "COORDINATE LEAKAGE DETECTED."
            )

        print(
            "Coordinate leakage: PASS"
        )

    else:

        print(
            "\nCoordinate columns not available."
        )

        print(
            "Coordinate leakage: NOT APPLICABLE"
        )

    return (
        source_overlap_count,
        sample_id_overlap_count,
        coordinate_overlap_count
    )


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

    n_pca = config["pca_components"]

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    # ============================================================
    # LOAD
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
            f"\nTraining file not found:\n{train_file}"
        )

    if not os.path.exists(
        test_file
    ):

        raise FileNotFoundError(
            f"\nTesting file not found:\n{test_file}"
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
    # PCA FEATURES
    # ============================================================

    print("\n" + "=" * 70)
    print("PCA FEATURES")
    print("=" * 70)

    pca_columns = [
        f"PC{i}"
        for i in range(
            1,
            n_pca + 1
        )
    ]

    print(
        "Number of PCA components:",
        n_pca
    )

    print(
        pca_columns
    )

    # ============================================================
    # PCA COLUMN VALIDATION
    # ============================================================

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

    if missing_train:

        raise ValueError(
            "Missing PCA columns in training:\n"
            f"{missing_train}"
        )

    if missing_test:

        raise ValueError(
            "Missing PCA columns in testing:\n"
            f"{missing_test}"
        )

    print(
        "\nAll PCA columns found."
    )

    # ============================================================
    # LABEL DETECTION
    # ============================================================

    train_label_column = find_label_column(
        train_df
    )

    test_label_column = find_label_column(
        test_df
    )

    print("\n" + "=" * 70)
    print("LABEL COLUMN DETECTION")
    print("=" * 70)

    print(
        "Training label column:",
        train_label_column
    )

    print(
        "Testing label column :",
        test_label_column
    )

    # ============================================================
    # ORIGINAL LABEL VALUES
    # ============================================================

    y_train_original = (
        train_df[
            train_label_column
        ]
        .astype(str)
        .to_numpy()
    )

    y_test_original = (
        test_df[
            test_label_column
        ]
        .astype(str)
        .to_numpy()
    )

    # ------------------------------------------------------------
    # Check class consistency
    # ------------------------------------------------------------

    train_classes_original = sorted(
        np.unique(
            y_train_original
        )
    )

    test_classes_original = sorted(
        np.unique(
            y_test_original
        )
    )

    print("\n" + "=" * 70)
    print("CLASS VALIDATION")
    print("=" * 70)

    print(
        "Training classes:",
        train_classes_original
    )

    print(
        "Testing classes :",
        test_classes_original
    )

    if (
        train_classes_original
        !=
        test_classes_original
    ):

        raise ValueError(
            "Training and testing class sets do not match."
        )

    n_classes = len(
        train_classes_original
    )

    print(
        "Number of classes:",
        n_classes
    )

    # ============================================================
    # LEAKAGE CHECK
    # ============================================================

    (
        source_overlap_count,
        sample_id_overlap_count,
        coordinate_overlap_count
    ) = perform_leakage_checks(
        train_df,
        test_df
    )

    # ============================================================
    # DATA MATRICES
    # ============================================================

    print("\n" + "=" * 70)
    print("DATA MATRICES")
    print("=" * 70)

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

    print(
        "Training matrix:",
        X_train.shape
    )

    print(
        "Testing matrix :",
        X_test.shape
    )

    print(
        "Training labels:",
        y_train_original.shape
    )

    print(
        "Testing labels :",
        y_test_original.shape
    )

    # ============================================================
    # NUMERICAL VALIDATION
    # ============================================================

    print("\n" + "=" * 70)
    print("NUMERICAL VALIDATION")
    print("=" * 70)

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
        "Training NaN:",
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
        or
        test_nan > 0
        or
        train_inf > 0
        or
        test_inf > 0
    ):

        raise ValueError(
            "NaN or Inf values detected."
        )

    print(
        "\nNumerical validation: PASS"
    )

    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        pd.Series(
            y_train_original
        ).value_counts(
            sort=False
        )
    )

    print("\n" + "=" * 70)
    print("TESTING CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        pd.Series(
            y_test_original
        ).value_counts(
            sort=False
        )
    )

    # ============================================================
    # LABEL ENCODING
    # ============================================================
    #
    # XGBoost requires classes to be:
    #
    # 0, 1, 2, ... n-1
    #
    # We therefore encode the original labels internally.
    #
    # The original labels are preserved separately.
    #
    # ============================================================

    print("\n" + "=" * 70)
    print("XGBOOST LABEL ENCODING")
    print("=" * 70)

    label_encoder = LabelEncoder()

    label_encoder.fit(
        y_train_original
    )

    y_train = (
        label_encoder.transform(
            y_train_original
        )
    )

    y_test = (
        label_encoder.transform(
            y_test_original
        )
    )

    encoded_classes = (
        label_encoder.classes_
    )

    print(
        "Original classes:"
    )

    print(
        list(encoded_classes)
    )

    print(
        "\nEncoded classes:"
    )

    print(
        list(
            range(n_classes)
        )
    )

    # ============================================================
    # TRAIN XGBOOST
    # ============================================================

    print("\n" + "=" * 70)
    print("TRAINING FINAL XGBOOST")
    print("=" * 70)

    print(
        "\nXGBoost parameters:"
    )

    print(
        "n_estimators     :",
        N_ESTIMATORS
    )

    print(
        "max_depth        :",
        MAX_DEPTH
    )

    print(
        "learning_rate    :",
        LEARNING_RATE
    )

    print(
        "subsample        :",
        SUBSAMPLE
    )

    print(
        "colsample_bytree :",
        COLSAMPLE_BYTREE
    )

    print(
        "random_state     :",
        RANDOM_STATE
    )

    print(
        "n_jobs           :",
        N_JOBS
    )

    print(
        "tree_method      : hist"
    )

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples :",
        len(X_test)
    )

    print(
        "Input features  :",
        X_train.shape[1]
    )

    print(
        "Number of classes:",
        n_classes
    )

    print(
        "\nFitting XGBoost..."
    )

    start_time = time.time()

    model = XGBClassifier(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        objective="multi:softmax",

        num_class=n_classes,

        eval_metric="mlogloss",

        random_state=RANDOM_STATE,

        n_jobs=N_JOBS,

        tree_method="hist"
    )

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time()
        -
        start_time
    )

    print(
        "\nXGBoost training completed."
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

    predicted_encoded = model.predict(
        X_test
    )

    prediction_time = (
        time.time()
        -
        prediction_start
    )

    predicted_encoded = (
        predicted_encoded.astype(
            np.int64
        )
    )

    # Convert back to original labels
    predictions_original = (
        label_encoder.inverse_transform(
            predicted_encoded
        )
    )

    print(
        "Predictions generated:",
        len(predictions_original)
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
            y_test_original,
            predictions_original
        )
        * 100
    )

    macro_precision = (
        precision_score(
            y_test_original,
            predictions_original,
            average="macro",
            zero_division=0
        )
        * 100
    )

    macro_recall = (
        recall_score(
            y_test_original,
            predictions_original,
            average="macro",
            zero_division=0
        )
        * 100
    )

    macro_f1 = (
        f1_score(
            y_test_original,
            predictions_original,
            average="macro",
            zero_division=0
        )
        * 100
    )

    # ============================================================
    # MAIN RESULTS
    # ============================================================

    print("\n" + "=" * 70)
    print(
        f"FINAL {dataset_name.upper()} XGBOOST RESULTS"
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

    report_text = classification_report(
        y_test_original,
        predictions_original,
        labels=list(
            encoded_classes
        ),
        target_names=list(
            encoded_classes
        ),
        zero_division=0
    )

    print(
        report_text
    )

    # ============================================================
    # PER-CLASS PERFORMANCE
    # ============================================================

    report_dict = classification_report(
        y_test_original,
        predictions_original,
        labels=list(
            encoded_classes
        ),
        target_names=list(
            encoded_classes
        ),
        output_dict=True,
        zero_division=0
    )

    per_class_rows = []

    for class_name in encoded_classes:

        class_name = str(
            class_name
        )

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

    per_class_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_final_xgboost_per_class.csv"
    )

    per_class_df.to_csv(
        per_class_file,
        index=False
    )

    print(
        "\nSaved per-class performance:"
    )

    print(
        per_class_file
    )

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        y_test_original,
        predictions_original,
        labels=list(
            encoded_classes
        )
    )

    print(
        cm
    )

    # CSV
    cm_csv = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_final_xgboost_confusion_matrix.csv"
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{x}"
            for x in encoded_classes
        ],
        columns=[
            f"Predicted_{x}"
            for x in encoded_classes
        ]
    )

    cm_df.to_csv(
        cm_csv
    )

    print(
        "\nSaved confusion matrix:"
    )

    print(
        cm_csv
    )

    # ============================================================
    # CONFUSION MATRIX GRAPH
    # ============================================================

    cm_png = os.path.join(
        graph_dir,
        f"{dataset_name.lower()}_final_xgboost_confusion_matrix.png"
    )

    plt.figure(
        figsize=(
            9,
            7
        )
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{dataset_name} XGBoost Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.xticks(
        range(n_classes),
        encoded_classes,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(n_classes),
        encoded_classes
    )

    for i in range(n_classes):

        for j in range(n_classes):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        cm_png,
        dpi=300
    )

    plt.close()

    print(
        "Saved confusion matrix graph:"
    )

    print(
        cm_png
    )

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    print("\n" + "=" * 70)
    print("XGBOOST FEATURE IMPORTANCE")
    print("=" * 70)

    feature_importance_df = pd.DataFrame({

        "feature":
            pca_columns,

        "importance":
            model.feature_importances_
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

    importance_csv = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_final_xgboost_feature_importance.csv"
    )

    feature_importance_df.to_csv(
        importance_csv,
        index=False
    )

    print(
        "\nSaved feature importance:"
    )

    print(
        importance_csv
    )

    # ============================================================
    # FEATURE IMPORTANCE GRAPH
    # ============================================================

    importance_png = os.path.join(
        graph_dir,
        f"{dataset_name.lower()}_final_xgboost_feature_importance.png"
    )

    plot_df = (
        feature_importance_df
        .head(
            min(
                20,
                len(feature_importance_df)
            )
        )
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(
            10,
            7
        )
    )

    plt.barh(
        plot_df["feature"],
        plot_df["importance"]
    )

    plt.xlabel(
        "XGBoost Feature Importance"
    )

    plt.ylabel(
        "PCA Component"
    )

    plt.title(
        f"{dataset_name} XGBoost Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        importance_png,
        dpi=300
    )

    plt.close()

    print(
        "Saved feature importance graph:"
    )

    print(
        importance_png
    )

    # ============================================================
    # SAVE PREDICTIONS
    # ============================================================

    prediction_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_final_xgboost_predictions.csv"
    )

    predictions_df = pd.DataFrame()

    # Preserve useful metadata if available
    metadata_columns = [

        "Sample_ID",

        "source_file",

        "row",

        "column",

        "class_name"
    ]

    for col in metadata_columns:

        if col in test_df.columns:

            predictions_df[
                col
            ] = test_df[
                col
            ].values

    # Original actual label
    predictions_df[
        "actual_label"
    ] = y_test_original

    # Predicted label
    predictions_df[
        "predicted_label"
    ] = predictions_original

    # Encoded prediction
    predictions_df[
        "predicted_encoded"
    ] = predicted_encoded

    # Correct / incorrect
    predictions_df[
        "correct"
    ] = (
        y_test_original
        ==
        predictions_original
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
        f"{dataset_name.lower()}_final_xgboost_metrics.csv"
    )

    metrics_df = pd.DataFrame([{

        "dataset":
            dataset_name,

        "model":
            "XGBoost",

        "pca_components":
            n_pca,

        "n_estimators":
            N_ESTIMATORS,

        "max_depth":
            MAX_DEPTH,

        "learning_rate":
            LEARNING_RATE,

        "subsample":
            SUBSAMPLE,

        "colsample_bytree":
            COLSAMPLE_BYTREE,

        "random_state":
            RANDOM_STATE,

        "training_samples":
            len(X_train),

        "testing_samples":
            len(X_test),

        "input_features":
            X_train.shape[1],

        "number_of_classes":
            n_classes,

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

        "source_file_overlap":
            source_overlap_count,

        "sample_id_overlap":
            sample_id_overlap_count,

        "coordinate_overlap":
            coordinate_overlap_count
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
    # FINAL RESULT
    # ============================================================

    print("\n" + "=" * 70)

    print(
        f"FINAL {dataset_name.upper()} XGBOOST COMPLETED"
    )

    print("=" * 70)

    print(
        f"""
Final test accuracy : {accuracy:.2f}%
Macro Precision     : {macro_precision:.2f}%
Macro Recall        : {macro_recall:.2f}%
Macro F1            : {macro_f1:.2f}%

PCA components      : {n_pca}

Training samples    : {len(X_train):,}
Testing samples     : {len(X_test):,}

Input features      : {X_train.shape[1]}
Number of classes   : {n_classes}

Training time       : {training_time / 60:.2f} minutes
Prediction time     : {prediction_time / 60:.2f} minutes

Source overlap      : {source_overlap_count}
Sample_ID overlap   : {sample_id_overlap_count}
Coordinate overlap  : {coordinate_overlap_count}

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

    del y_train_original
    del y_test_original

    del y_train
    del y_test

    del predicted_encoded
    del predictions_original

    del model
    del label_encoder

    gc.collect()

    return {

        "dataset":
            dataset_name,

        "model":
            "XGBoost",

        "pca_components":
            n_pca,

        "training_samples":
            metrics_df.loc[
                0,
                "training_samples"
            ],

        "testing_samples":
            metrics_df.loc[
                0,
                "testing_samples"
            ],

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

        "source_file_overlap":
            source_overlap_count,

        "sample_id_overlap":
            sample_id_overlap_count,

        "coordinate_overlap":
            coordinate_overlap_count
    }


# ================================================================
# MAIN
# ================================================================

print("\n" + "=" * 70)
print("MULTI-CROP FINAL XGBOOST CLASSIFICATION")
print("=" * 70)

print(
"""
THIRD CLASSIFIER COMPARISON

Already completed:
    Random Forest
    SVM

Current experiment:
    XGBoost

Existing leakage-safe train/test datasets
will be used without creating a new split.

FINAL PCA:

    Tomato   : 20 PCs
    Papaya   : 90 PCs
    Eggplant : 30 PCs
    Cabbage  : 30 PCs
"""
)


# ================================================================
# VERSION
# ================================================================

try:

    import xgboost

    print(
        "\nXGBoost version:",
        xgboost.__version__
    )

except Exception as e:

    raise ImportError(
        "XGBoost is not installed."
    ) from e


# ================================================================
# RUN ALL CROPS
# ================================================================

results = []

for dataset_name, config in DATASETS.items():

    result = process_dataset(
        dataset_name,
        config
    )

    results.append(
        result
    )


# ================================================================
# COMBINED SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("FINAL MULTI-CROP XGBOOST SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(
    results
)

print(
    results_df.to_string(
        index=False
    )
)

summary_file = os.path.join(
    PROCESSED_DIR,
    "multi_crop_final_xgboost_summary.csv"
)

results_df.to_csv(
    summary_file,
    index=False
)

print(
    "\nCombined summary saved:"
)

print(
    summary_file
)


# ================================================================
# FINAL
# ================================================================

print("\n" + "=" * 70)
print("MULTI-CROP FINAL XGBOOST CLASSIFICATION COMPLETED")
print("=" * 70)

print(
"""
THIRD CLASSIFIER EXPERIMENT COMPLETED.

Final comparison:

    Random Forest
          vs
        SVM
          vs
       XGBoost

Crops:

    Tomato
    Papaya
    Eggplant
    Cabbage

Metrics:

    Accuracy
    Macro Precision
    Macro Recall
    Macro F1
    Per-class performance
    Training time
    Prediction time
    Confusion matrix
    Feature importance
"""
)

print("=" * 70)