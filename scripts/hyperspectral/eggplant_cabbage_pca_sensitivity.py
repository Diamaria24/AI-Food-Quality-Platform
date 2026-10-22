# ================================================================
# EGGPLANT + CABBAGE HSI PCA COMPONENT SENSITIVITY
# ================================================================
#
# Purpose:
#   Determine the useful number of PCA components for
#   Eggplant and Cabbage HSI classification.
#
# Method:
#
#   Spatially separated train/test data
#           ↓
#   Spectral smoothing
#           ↓
#   Training-derived standardization
#           ↓
#   PCA fitted ONLY on training data
#           ↓
#   Transform train + test
#           ↓
#   Random Forest classification
#           ↓
#   Compare different PCA component counts
#
# Tested:
#   2, 10, 20, 30, 40, 50, 60, 70, 80, 90
#
# IMPORTANT:
#   Test data is NEVER used to fit StandardScaler or PCA.
#
# ================================================================

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
        "npz": os.path.join(
            PROCESSED_DIR,
            "eggplant_classification",
            "eggplant_spatial_classification.npz"
        ),

        "wavelengths": os.path.join(
            PROCESSED_DIR,
            "eggplant_classification",
            "eggplant_wavelengths.csv"
        ),

        "output": os.path.join(
            PROCESSED_DIR,
            "eggplant_classification",
            "pca_sensitivity"
        ),

        "graphs": os.path.join(
            GRAPH_DIR,
            "eggplant_classification"
        )
    },

    "Cabbage": {
        "npz": os.path.join(
            PROCESSED_DIR,
            "cabbage_classification",
            "cabbage_spatial_classification.npz"
        ),

        "wavelengths": os.path.join(
            PROCESSED_DIR,
            "cabbage_classification",
            "cabbage_wavelengths.csv"
        ),

        "output": os.path.join(
            PROCESSED_DIR,
            "cabbage_classification",
            "pca_sensitivity"
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

RANDOM_STATE = 42

# PCA component values to test
PCA_COMPONENTS = [
    2,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90
]

# Random Forest settings for sensitivity experiments.
#
# We use 150 trees for the repeated sensitivity experiments.
# The final selected PCA model can later be trained with 200 trees.
RF_TREES = 150

# Savitzky-Golay smoothing
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2


# ================================================================
# CREATE OUTPUT DIRECTORIES
# ================================================================

for dataset_name, config in DATASETS.items():

    os.makedirs(
        config["output"],
        exist_ok=True
    )

    os.makedirs(
        config["graphs"],
        exist_ok=True
    )


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("EGGPLANT + CABBAGE HSI PCA COMPONENT SENSITIVITY")
print("=" * 70)

print("""
METHODOLOGY

1. Load spatially separated train/test spectra.
2. Apply spectral smoothing.
3. Fit StandardScaler ONLY on training data.
4. Transform testing data using training parameters.
5. Fit PCA ONLY on training data.
6. Transform testing data using the training PCA.
7. Train Random Forest.
8. Evaluate on completely unseen spatial test pixels.

PCA COMPONENTS:
2, 10, 20, 30, 40, 50, 60, 70, 80, 90
""")


# ================================================================
# PREPROCESSING FUNCTION
# ================================================================

def preprocess_spectra(
    X_train,
    X_test
):

    print(
        "\nApplying Savitzky-Golay spectral smoothing..."
    )

    # ------------------------------------------------------------
    # Smooth along spectral-band axis
    # ------------------------------------------------------------

    X_train_smooth = savgol_filter(
        X_train,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER,
        axis=1
    ).astype(
        np.float32
    )

    X_test_smooth = savgol_filter(
        X_test,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER,
        axis=1
    ).astype(
        np.float32
    )

    print(
        "Smoothing completed."
    )

    # ------------------------------------------------------------
    # Training-derived StandardScaler
    # ------------------------------------------------------------

    print(
        "\nFitting StandardScaler using "
        "training data only..."
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_smooth
    ).astype(
        np.float32
    )

    print(
        "Transforming testing data using "
        "training-derived parameters..."
    )

    X_test_scaled = scaler.transform(
        X_test_smooth
    ).astype(
        np.float32
    )

    print(
        "Training standardization completed."
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
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
        f"{dataset_name.upper()} PCA SENSITIVITY"
    )
    print("=" * 70)

    # ============================================================
    # FILE CHECK
    # ============================================================

    npz_path = config["npz"]

    if not os.path.exists(
        npz_path
    ):

        raise FileNotFoundError(
            f"\nNPZ file not found:\n{npz_path}"
        )

    print(
        "\nInput file:"
    )

    print(
        npz_path
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)

    data = np.load(
        npz_path
    )

    X_train = data[
        "X_train"
    ]

    X_test = data[
        "X_test"
    ]

    y_train = data[
        "y_train"
    ]

    y_test = data[
        "y_test"
    ]

    print(
        "\nTraining shape:",
        X_train.shape
    )

    print(
        "Testing shape:",
        X_test.shape
    )

    print(
        "Training labels:",
        y_train.shape
    )

    print(
        "Testing labels:",
        y_test.shape
    )

    # ============================================================
    # COORDINATES
    # ============================================================

    rows_train = data[
        "rows_train"
    ]

    cols_train = data[
        "cols_train"
    ]

    rows_test = data[
        "rows_test"
    ]

    cols_test = data[
        "cols_test"
    ]

    # ============================================================
    # VALIDATION
    # ============================================================

    print("\n" + "-" * 70)
    print("DATA VALIDATION")
    print("-" * 70)

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
        or test_nan > 0
        or train_inf > 0
        or test_inf > 0
    ):

        raise ValueError(
            "Invalid numerical values detected."
        )

    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    print(
        "\nTraining:"
    )

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    print(
        "\nTesting:"
    )

    print(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
    )

    # ============================================================
    # COORDINATE LEAKAGE CHECK
    # ============================================================

    print("\n" + "-" * 70)
    print("SPATIAL COORDINATE LEAKAGE CHECK")
    print("-" * 70)

    train_coordinates = set(
        zip(
            rows_train.tolist(),
            cols_train.tolist()
        )
    )

    test_coordinates = set(
        zip(
            rows_test.tolist(),
            cols_test.tolist()
        )
    )

    coordinate_overlap = (
        train_coordinates
        &
        test_coordinates
    )

    print(
        "Training coordinates:",
        len(train_coordinates)
    )

    print(
        "Testing coordinates:",
        len(test_coordinates)
    )

    print(
        "Overlapping coordinates:",
        len(coordinate_overlap)
    )

    if len(coordinate_overlap) != 0:

        raise ValueError(
            "Spatial coordinate leakage detected."
        )

    print(
        "Coordinate leakage check: PASS"
    )

    # ============================================================
    # PREPROCESSING
    # ============================================================

    X_train_processed, X_test_processed, scaler = (
        preprocess_spectra(
            X_train,
            X_test
        )
    )

    # Original arrays no longer needed
    del X_train
    del X_test

    gc.collect()

    # ============================================================
    # PCA SENSITIVITY
    # ============================================================

    results = []

    max_components = min(
        X_train_processed.shape[0],
        X_train_processed.shape[1]
    )

    print("\n" + "=" * 70)
    print(
        "PCA SENSITIVITY EXPERIMENTS"
    )
    print("=" * 70)

    print(
        "\nMaximum possible PCA components:",
        max_components
    )

    print(
        "Components tested:",
        PCA_COMPONENTS
    )

    # ============================================================
    # RUN EACH PCA EXPERIMENT
    # ============================================================

    for n_components in PCA_COMPONENTS:

        print("\n")
        print("-" * 70)
        print(
            f"EXPERIMENT: {n_components} PCA COMPONENTS"
        )
        print("-" * 70)

        if n_components > max_components:

            print(
                f"Skipping {n_components} PCs: "
                "greater than maximum."
            )

            continue

        # --------------------------------------------------------
        # FIT PCA ONLY ON TRAINING DATA
        # --------------------------------------------------------

        print(
            "\nFitting PCA using training data only..."
        )

        pca = PCA(
            n_components=n_components,
            random_state=RANDOM_STATE
        )

        X_train_pca = pca.fit_transform(
            X_train_processed
        ).astype(
            np.float32
        )

        print(
            "Transforming testing data "
            "using training PCA..."
        )

        X_test_pca = pca.transform(
            X_test_processed
        ).astype(
            np.float32
        )

        variance_retained = (
            np.sum(
                pca.explained_variance_ratio_
            )
            * 100
        )

        print(
            "\nTraining PCA shape:",
            X_train_pca.shape
        )

        print(
            "Testing PCA shape :",
            X_test_pca.shape
        )

        print(
            "Variance retained :",
            f"{variance_retained:.4f}%"
        )

        # --------------------------------------------------------
        # RANDOM FOREST
        # --------------------------------------------------------

        print(
            "\nTraining Random Forest..."
        )

        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        rf.fit(
            X_train_pca,
            y_train
        )

        print(
            "Random Forest training completed."
        )

        # --------------------------------------------------------
        # PREDICTIONS
        # --------------------------------------------------------

        predictions = rf.predict(
            X_test_pca
        )

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
            * 100
        )

        precision = (
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
            * 100
        )

        recall = (
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
            * 100
        )

        f1 = (
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
            * 100
        )

        print(
            "\nRESULTS"
        )

        print(
            "Accuracy        :",
            f"{accuracy:.2f}%"
        )

        print(
            "Macro Precision :",
            f"{precision:.2f}%"
        )

        print(
            "Macro Recall    :",
            f"{recall:.2f}%"
        )

        print(
            "Macro F1        :",
            f"{f1:.2f}%"
        )

        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------

        results.append({

            "PCA Components":
                n_components,

            "Variance Retained (%)":
                variance_retained,

            "Accuracy (%)":
                accuracy,

            "Macro Precision (%)":
                precision,

            "Macro Recall (%)":
                recall,

            "Macro F1 (%)":
                f1
        })

        # --------------------------------------------------------
        # CLEAN MEMORY
        # --------------------------------------------------------

        del pca
        del rf
        del X_train_pca
        del X_test_pca
        del predictions

        gc.collect()

    # ============================================================
    # RESULTS DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(
        results
    )

    # ============================================================
    # FIND BEST RESULT
    # ============================================================

    best_index = (
        results_df[
            "Accuracy (%)"
        ].idxmax()
    )

    best_result = results_df.loc[
        best_index
    ]

    best_components = int(
        best_result[
            "PCA Components"
        ]
    )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results_file = os.path.join(
        config["output"],
        f"{dataset_name.lower()}_pca_sensitivity_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("PCA SENSITIVITY RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved results:"
    )

    print(
        results_file
    )

    # ============================================================
    # ACCURACY GRAPH
    # ============================================================

    accuracy_graph = os.path.join(
        config["graphs"],
        f"{dataset_name.lower()}_pca_component_accuracy.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results_df[
            "PCA Components"
        ],
        results_df[
            "Accuracy (%)"
        ],
        marker="o"
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Random Forest Accuracy (%)"
    )

    plt.title(
        f"{dataset_name} HSI "
        "PCA Component Sensitivity"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        accuracy_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # VARIANCE GRAPH
    # ============================================================

    variance_graph = os.path.join(
        config["graphs"],
        f"{dataset_name.lower()}_pca_component_variance.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results_df[
            "PCA Components"
        ],
        results_df[
            "Variance Retained (%)"
        ],
        marker="o"
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Variance Retained (%)"
    )

    plt.title(
        f"{dataset_name} HSI "
        "PCA Variance Retention"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        variance_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # BEST RESULT
    # ============================================================

    print("\n" + "=" * 70)
    print(
        f"BEST {dataset_name.upper()} PCA RESULT"
    )
    print("=" * 70)

    print(
        "\nPCA components:",
        best_components
    )

    print(
        "Variance retained:",
        f"{best_result['Variance Retained (%)']:.4f}%"
    )

    print(
        "Accuracy:",
        f"{best_result['Accuracy (%)']:.2f}%"
    )

    print(
        "Macro Precision:",
        f"{best_result['Macro Precision (%)']:.2f}%"
    )

    print(
        "Macro Recall:",
        f"{best_result['Macro Recall (%)']:.2f}%"
    )

    print(
        "Macro F1:",
        f"{best_result['Macro F1 (%)']:.2f}%"
    )

    print(
        "\nAccuracy graph:"
    )

    print(
        accuracy_graph
    )

    print(
        "\nVariance graph:"
    )

    print(
        variance_graph
    )

    # ============================================================
    # SAVE BEST COMPONENT INFORMATION
    # ============================================================

    best_file = os.path.join(
        config["output"],
        f"{dataset_name.lower()}_best_pca_components.csv"
    )

    pd.DataFrame([
        {
            "best_pca_components":
                best_components,

            "variance_retained_percent":
                best_result[
                    "Variance Retained (%)"
                ],

            "accuracy_percent":
                best_result[
                    "Accuracy (%)"
                ],

            "macro_precision_percent":
                best_result[
                    "Macro Precision (%)"
                ],

            "macro_recall_percent":
                best_result[
                    "Macro Recall (%)"
                ],

            "macro_f1_percent":
                best_result[
                    "Macro F1 (%)"
                ]
        }
    ]).to_csv(
        best_file,
        index=False
    )

    print(
        "\nSaved best PCA information:"
    )

    print(
        best_file
    )

    # ============================================================
    # CLEAN
    # ============================================================

    del X_train_processed
    del X_test_processed
    del y_train
    del y_test
    del scaler
    del data

    gc.collect()

    # ============================================================
    # FINAL
    # ============================================================

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} PCA SENSITIVITY COMPLETED"
    )
    print("=" * 70)

    return {
        "dataset": dataset_name,
        "best_components": best_components,
        "accuracy": float(
            best_result["Accuracy (%)"]
        ),
        "macro_f1": float(
            best_result["Macro F1 (%)"]
        ),
        "variance": float(
            best_result[
                "Variance Retained (%)"
            ]
        )
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
print("EGGPLANT + CABBAGE PCA SENSITIVITY SUMMARY")
print("=" * 70)

combined_results = pd.DataFrame([

    eggplant_result,
    cabbage_result
])

print(
    combined_results.to_string(
        index=False
    )
)

combined_file = os.path.join(
    PROCESSED_DIR,
    "eggplant_cabbage_pca_sensitivity_summary.csv"
)

combined_results.to_csv(
    combined_file,
    index=False
)

print(
    "\nSaved combined PCA summary:"
)

print(
    combined_file
)


# ================================================================
# FINAL MESSAGE
# ================================================================

print("\n" + "=" * 70)
print("EGGPLANT + CABBAGE PCA SENSITIVITY COMPLETED")
print("=" * 70)

print("""
NEXT STEP:

For each dataset we will take the best PCA component count
identified here and create the final PCA classification dataset.

Then:

        Best PCA
           ↓
    Final Random Forest
           ↓
        Final SVM
           ↓
    RF vs SVM comparison
""")

print("=" * 70)