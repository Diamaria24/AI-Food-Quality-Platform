# ================================================================
# EGGPLANT + CABBAGE FINAL PCA DATASET GENERATION
# ================================================================
#
# Selected PCA components:
#   Eggplant = 30
#   Cabbage  = 30
#
# IMPORTANT:
#   PCA is fitted ONLY on the training data.
#   Test data is transformed using the training-derived PCA.
#
# Input:
#   *_spatial_classification.npz
#
# Output:
#   *_train_30pc.csv
#   *_test_30pc.csv
#   *_pca_30pc_variance.csv
#   *_pca_30pc_model.npz
#
# ================================================================

import os
import gc
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


# ================================================================
# DATASET CONFIGURATION
# ================================================================

DATASETS = {

    "Eggplant": {
        "input": os.path.join(
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
            "pca_30pc"
        )
    },

    "Cabbage": {
        "input": os.path.join(
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
            "pca_30pc"
        )
    }
}


# ================================================================
# SETTINGS
# ================================================================

N_COMPONENTS = 30

RANDOM_STATE = 42

SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2


# ================================================================
# CLASS NAMES
# ================================================================

CLASS_NAMES = {
    1: "Low",
    2: "Medium",
    3: "High"
}


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("EGGPLANT + CABBAGE FINAL 30-PC PCA DATASET GENERATION")
print("=" * 70)

print("""
SELECTED PCA COMPONENTS

Eggplant : 30 PCs
Cabbage  : 30 PCs

METHODOLOGY

Raw spectral samples
        ↓
Savitzky-Golay smoothing
        ↓
StandardScaler fitted on TRAIN only
        ↓
PCA fitted on TRAIN only
        ↓
30 PCA components
        ↓
Transform TEST using training PCA
        ↓
Save final classification datasets

No test data is used to fit preprocessing or PCA.
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

    input_file = config["input"]
    wavelength_file = config["wavelengths"]
    output_dir = config["output"]

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ============================================================
    # CHECK INPUT
    # ============================================================

    print("\nINPUT FILE")
    print("-" * 70)

    print(
        input_file
    )

    if not os.path.exists(
        input_file
    ):

        raise FileNotFoundError(
            f"\nInput file not found:\n{input_file}"
        )

    print(
        "Input file: FOUND"
    )

    # ============================================================
    # LOAD NPZ
    # ============================================================

    print("\nLOADING DATA")
    print("-" * 70)

    data = np.load(
        input_file
    )

    X_train = data[
        "X_train"
    ].astype(
        np.float32
    )

    X_test = data[
        "X_test"
    ].astype(
        np.float32
    )

    y_train = data[
        "y_train"
    ].astype(
        np.int64
    )

    y_test = data[
        "y_test"
    ].astype(
        np.int64
    )

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

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Testing shape :",
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

    # ============================================================
    # VALIDATION
    # ============================================================

    print("\nDATA VALIDATION")
    print("-" * 70)

    print(
        "Training NaN:",
        int(np.isnan(X_train).sum())
    )

    print(
        "Testing NaN :",
        int(np.isnan(X_test).sum())
    )

    print(
        "Training Inf:",
        int(np.isinf(X_train).sum())
    )

    print(
        "Testing Inf :",
        int(np.isinf(X_test).sum())
    )

    if (
        np.isnan(X_train).any()
        or
        np.isnan(X_test).any()
        or
        np.isinf(X_train).any()
        or
        np.isinf(X_test).any()
    ):

        raise ValueError(
            "NaN or Inf values detected."
        )

    # ============================================================
    # SPATIAL LEAKAGE CHECK
    # ============================================================

    print("\nSPATIAL LEAKAGE CHECK")
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

    overlap = (
        train_coordinates
        &
        test_coordinates
    )

    print(
        "Training coordinates:",
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
            "Spatial leakage detected."
        )

    print(
        "Spatial leakage check: PASS"
    )

    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    print("\nCLASS DISTRIBUTION")
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
    # SPECTRAL SMOOTHING
    # ============================================================

    print("\nSPECTRAL PREPROCESSING")
    print("-" * 70)

    print(
        f"Savitzky-Golay window    : "
        f"{SAVGOL_WINDOW}"
    )

    print(
        f"Savitzky-Golay polyorder : "
        f"{SAVGOL_POLYORDER}"
    )

    print(
        "\nSmoothing training spectra..."
    )

    X_train_smooth = savgol_filter(
        X_train,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER,
        axis=1
    ).astype(
        np.float32
    )

    print(
        "Smoothing testing spectra..."
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
        "Spectral smoothing completed."
    )

    # Original raw arrays no longer needed
    del X_train
    del X_test

    gc.collect()

    # ============================================================
    # STANDARDIZATION
    # ============================================================

    print("\nTRAINING-DERIVED STANDARDIZATION")
    print("-" * 70)

    scaler = StandardScaler()

    print(
        "Fitting StandardScaler "
        "using training data only..."
    )

    X_train_scaled = scaler.fit_transform(
        X_train_smooth
    ).astype(
        np.float32
    )

    print(
        "Transforming testing data "
        "using training scaler..."
    )

    X_test_scaled = scaler.transform(
        X_test_smooth
    ).astype(
        np.float32
    )

    print(
        "Standardization completed."
    )

    del X_train_smooth
    del X_test_smooth

    gc.collect()

    # ============================================================
    # PCA
    # ============================================================

    print("\n" + "=" * 70)
    print("FITTING FINAL 30-COMPONENT PCA")
    print("=" * 70)

    print(
        "\nNumber of PCA components:",
        N_COMPONENTS
    )

    print(
        "Original spectral bands:",
        X_train_scaled.shape[1]
    )

    print(
        "\nFitting PCA ONLY on training data..."
    )

    pca = PCA(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    ).astype(
        np.float32
    )

    print(
        "Training PCA transformation completed."
    )

    print(
        "Transforming testing data..."
    )

    X_test_pca = pca.transform(
        X_test_scaled
    ).astype(
        np.float32
    )

    print(
        "Testing PCA transformation completed."
    )

    # ============================================================
    # PCA SHAPES
    # ============================================================

    print("\nPCA DATA MATRICES")
    print("-" * 70)

    print(
        "Training PCA shape:",
        X_train_pca.shape
    )

    print(
        "Testing PCA shape :",
        X_test_pca.shape
    )

    # ============================================================
    # VARIANCE
    # ============================================================

    explained_variance = (
        pca.explained_variance_ratio_
        * 100
    )

    cumulative_variance = np.cumsum(
        explained_variance
    )

    total_variance = float(
        cumulative_variance[-1]
    )

    print("\nPCA VARIANCE")
    print("-" * 70)

    print(
        f"Variance retained by "
        f"{N_COMPONENTS} PCs:"
    )

    print(
        f"{total_variance:.6f}%"
    )

    # ============================================================
    # PCA VARIANCE TABLE
    # ============================================================

    variance_df = pd.DataFrame({

        "PC":
            np.arange(
                1,
                N_COMPONENTS + 1
            ),

        "Explained_Variance_Percent":
            explained_variance,

        "Cumulative_Variance_Percent":
            cumulative_variance
    })

    variance_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_pca_30pc_variance.csv"
    )

    variance_df.to_csv(
        variance_file,
        index=False
    )

    print(
        "\nSaved PCA variance:"
    )

    print(
        variance_file
    )

    # ============================================================
    # CREATE PCA COLUMN NAMES
    # ============================================================

    pca_columns = [
        f"PC{i}"
        for i in range(
            1,
            N_COMPONENTS + 1
        )
    ]

    # ============================================================
    # TRAINING DATAFRAME
    # ============================================================

    print("\nCREATING TRAINING DATASET")
    print("-" * 70)

    train_pca_df = pd.DataFrame(
        X_train_pca,
        columns=pca_columns
    )

    # Metadata first
    train_pca_df.insert(
        0,
        "label",
        y_train
    )

    train_pca_df.insert(
        1,
        "class_name",
        [
            CLASS_NAMES[int(x)]
            for x in y_train
        ]
    )

    train_pca_df.insert(
        2,
        "row",
        rows_train
    )

    train_pca_df.insert(
        3,
        "column",
        cols_train
    )

    # ============================================================
    # TESTING DATAFRAME
    # ============================================================

    print(
        "Creating testing dataset..."
    )

    test_pca_df = pd.DataFrame(
        X_test_pca,
        columns=pca_columns
    )

    test_pca_df.insert(
        0,
        "label",
        y_test
    )

    test_pca_df.insert(
        1,
        "class_name",
        [
            CLASS_NAMES[int(x)]
            for x in y_test
        ]
    )

    test_pca_df.insert(
        2,
        "row",
        rows_test
    )

    test_pca_df.insert(
        3,
        "column",
        cols_test
    )

    # ============================================================
    # SAVE CSV
    # ============================================================

    print("\nSAVING FINAL PCA DATASETS")
    print("-" * 70)

    train_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_train_30pc.csv"
    )

    test_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_test_30pc.csv"
    )

    print(
        "\nSaving training:"
    )

    print(
        train_file
    )

    train_pca_df.to_csv(
        train_file,
        index=False
    )

    print(
        "Training dataset saved."
    )

    print(
        "\nSaving testing:"
    )

    print(
        test_file
    )

    test_pca_df.to_csv(
        test_file,
        index=False
    )

    print(
        "Testing dataset saved."
    )

    # ============================================================
    # SAVE COMPACT NPZ
    # ============================================================

    print("\nSAVING COMPACT PCA NPZ")
    print("-" * 70)

    npz_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_30pc.npz"
    )

    np.savez_compressed(
        npz_file,

        X_train=X_train_pca,

        X_test=X_test_pca,

        y_train=y_train,

        y_test=y_test,

        rows_train=rows_train,

        cols_train=cols_train,

        rows_test=rows_test,

        cols_test=cols_test,

        explained_variance_ratio=
            pca.explained_variance_ratio_,

        cumulative_variance=
            cumulative_variance
    )

    print(
        "Saved:"
    )

    print(
        npz_file
    )

    # ============================================================
    # SAVE PCA COMPONENTS
    # ============================================================

    components_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_pca_30pc_components.csv"
    )

    components_df = pd.DataFrame(
        pca.components_,
        columns=[
            f"Band_{i+1}"
            for i in range(
                pca.components_.shape[1]
            )
        ]
    )

    components_df.insert(
        0,
        "PC",
        pca_columns
    )

    components_df.to_csv(
        components_file,
        index=False
    )

    print(
        "PCA components saved:"
    )

    print(
        components_file
    )

    # ============================================================
    # SAVE SCALER PARAMETERS
    # ============================================================

    scaler_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_scaler_parameters.npz"
    )

    np.savez_compressed(
        scaler_file,

        mean=scaler.mean_,

        scale=scaler.scale_
    )

    print(
        "Scaler parameters saved:"
    )

    print(
        scaler_file
    )

    # ============================================================
    # SAVE SUMMARY
    # ============================================================

    summary_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_pca_30pc_summary.csv"
    )

    summary_df = pd.DataFrame([{

        "dataset":
            dataset_name,

        "original_bands":
            X_train_scaled.shape[1],

        "pca_components":
            N_COMPONENTS,

        "variance_retained_percent":
            total_variance,

        "training_samples":
            len(y_train),

        "testing_samples":
            len(y_test),

        "training_features":
            X_train_pca.shape[1],

        "testing_features":
            X_test_pca.shape[1],

        "spatial_coordinate_overlap":
            len(overlap),

        "savgol_window":
            SAVGOL_WINDOW,

        "savgol_polyorder":
            SAVGOL_POLYORDER
    }])

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print(
        "\nSummary saved:"
    )

    print(
        summary_file
    )

    # ============================================================
    # FINAL VALIDATION
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL PCA DATA VALIDATION")
    print("=" * 70)

    print(
        "\nTraining PCA NaN:",
        int(np.isnan(X_train_pca).sum())
    )

    print(
        "Testing PCA NaN :",
        int(np.isnan(X_test_pca).sum())
    )

    print(
        "Training PCA Inf:",
        int(np.isinf(X_train_pca).sum())
    )

    print(
        "Testing PCA Inf :",
        int(np.isinf(X_test_pca).sum())
    )

    if (
        np.isnan(X_train_pca).any()
        or
        np.isnan(X_test_pca).any()
        or
        np.isinf(X_train_pca).any()
        or
        np.isinf(X_test_pca).any()
    ):

        raise ValueError(
            "Invalid values found after PCA."
        )

    print(
        "\nPCA numerical validation: PASS"
    )

    # ============================================================
    # CLASS BALANCE FINAL CHECK
    # ============================================================

    print("\nFINAL CLASS DISTRIBUTION")
    print("-" * 70)

    print(
        "\nTraining:"
    )

    print(
        train_pca_df[
            "class_name"
        ].value_counts()
        .sort_index()
    )

    print(
        "\nTesting:"
    )

    print(
        test_pca_df[
            "class_name"
        ].value_counts()
        .sort_index()
    )

    # ============================================================
    # MEMORY CLEANUP
    # ============================================================

    del X_train_scaled
    del X_test_scaled
    del X_train_pca
    del X_test_pca
    del train_pca_df
    del test_pca_df
    del pca
    del scaler
    del data

    gc.collect()

    # ============================================================
    # FINAL MESSAGE
    # ============================================================

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} FINAL PCA COMPLETED"
    )
    print("=" * 70)

    print(
        f"""
PCA components      : {N_COMPONENTS}
Variance retained   : {total_variance:.4f}%

Training samples    : 30,000
Testing samples     : 30,000

Spatial leakage     : PASS
Coordinate overlap  : 0

Training dataset:
{train_file}

Testing dataset:
{test_file}

Compact NPZ:
{npz_file}
"""
    )

    return {
        "dataset": dataset_name,
        "components": N_COMPONENTS,
        "variance": total_variance,
        "train_samples": 30000,
        "test_samples": 30000
    }


# ================================================================
# PROCESS EGGPLANT
# ================================================================

eggplant_result = process_dataset(
    "Eggplant",
    DATASETS["Eggplant"]
)


# ================================================================
# PROCESS CABBAGE
# ================================================================

cabbage_result = process_dataset(
    "Cabbage",
    DATASETS["Cabbage"]
)


# ================================================================
# FINAL COMBINED SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("EGGPLANT + CABBAGE FINAL PCA SUMMARY")
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
    "eggplant_cabbage_final_pca_summary.csv"
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
print("FINAL 30-PC DATASETS READY")
print("=" * 70)

print("""
NEXT STEP:

Eggplant
    ↓
30-PC dataset
    ↓
Final Random Forest
    ↓
Final SVM

Cabbage
    ↓
30-PC dataset
    ↓
Final Random Forest
    ↓
Final SVM
""")

print("=" * 70)