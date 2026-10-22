"""
papaya_pca_sensitivity.py

REAL PAPAYA HSI PCA COMPONENT SENSITIVITY EXPERIMENT

Dataset:
    Garillos-Manliguez Hyperspectral Images of Papaya

Pipeline:
    Papaya ENVI HDR files
            ↓
    Load hyperspectral cubes
            ↓
    Match HDR files to CSV labels
            ↓
    Remove invalid spectra
            ↓
    Spectral scaling
            ↓
    Savitzky-Golay smoothing
            ↓
    Sample spectra from each HDR
            ↓
    Sample_ID based train/test separation
            ↓
    Training-derived mean centering
            ↓
    PCA fitted ONLY on training data
            ↓
    Transform test data using training PCA
            ↓
    Test multiple PCA component counts
            ↓
    Random Forest classification
            ↓
    Accuracy / Precision / Recall / Macro F1
            ↓
    Select best PCA configuration

IMPORTANT:
    PCA is NEVER fitted using test data.

    Sample_ID is treated as the physical-sample grouping variable.
    Therefore the same Sample_ID cannot appear in both training
    and testing data.
"""

import os
import glob
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = (
    r"D:\Projects\AI-Food-Quality-Platform"
)


PAPAYA_RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "papaya_hsi"
)


PAPAYA_CLASSIFICATION_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "papaya_classification"
)


OUTPUT_DIR = os.path.join(
    PAPAYA_CLASSIFICATION_DIR,
    "pca_sensitivity"
)


GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "papaya_classification"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# ======================================================================
# TRAIN / TEST SPLITS
# ======================================================================

TRAIN_SPLIT = os.path.join(
    PAPAYA_CLASSIFICATION_DIR,
    "papaya_train_split.csv"
)


TEST_SPLIT = os.path.join(
    PAPAYA_CLASSIFICATION_DIR,
    "papaya_test_split.csv"
)


# ======================================================================
# EXPERIMENT CONFIGURATION
# ======================================================================

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# IMPORTANT CHANGE:
#
# Previous experiment stopped at 50 PCs.
#
# This experiment goes to 100 PCs because accuracy was still improving
# at 50 PCs.
# ----------------------------------------------------------------------

MAX_PCA_COMPONENTS = 100


PCA_COMPONENTS_TO_TEST = [
    2,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100
]


# ----------------------------------------------------------------------
# Number of pixels sampled from each HDR cube.
#
# This keeps memory and computation manageable.
# ----------------------------------------------------------------------

SAMPLES_PER_CUBE = 200


# ----------------------------------------------------------------------
# Spectral preprocessing
# ----------------------------------------------------------------------

SAVGOL_WINDOW = 7

SAVGOL_POLYORDER = 2


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------

RF_TREES = 200


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("PAPAYA HSI EXTENDED PCA COMPONENT SENSITIVITY")
print("=" * 70)


print()
print("Dataset directory:")
print(PAPAYA_RAW_DIR)


print()
print("Maximum PCA components allowed:",
      MAX_PCA_COMPONENTS)


print()
print("PCA components to test:")
print(PCA_COMPONENTS_TO_TEST)


print()
print("=" * 70)
print("METHODOLOGY")
print("=" * 70)


print()
print("1. Training and testing are separated by Sample_ID.")


print(
    "2. PCA is fitted ONLY on training spectra."
)


print(
    "3. Test spectra are transformed using the training PCA."
)


print(
    "4. Random Forest is trained separately for each PCA size."
)


print(
    "5. The best configuration is selected using test accuracy."
)


# ======================================================================
# CHECK INPUTS
# ======================================================================

if not os.path.exists(
    PAPAYA_RAW_DIR
):

    raise FileNotFoundError(
        f"\nPapaya dataset directory not found:\n"
        f"{PAPAYA_RAW_DIR}"
    )


if not os.path.exists(
    TRAIN_SPLIT
):

    raise FileNotFoundError(
        f"\nTraining split not found:\n"
        f"{TRAIN_SPLIT}\n\n"
        "Run create_papaya_group_split.py first."
    )


if not os.path.exists(
    TEST_SPLIT
):

    raise FileNotFoundError(
        f"\nTesting split not found:\n"
        f"{TEST_SPLIT}\n\n"
        "Run create_papaya_group_split.py first."
    )


# ======================================================================
# LOAD TRAIN / TEST CSV SPLITS
# ======================================================================

print()
print("=" * 70)
print("LOADING PAPAYA TRAIN / TEST SPLITS")
print("=" * 70)


train_df = pd.read_csv(
    TRAIN_SPLIT
)


test_df = pd.read_csv(
    TEST_SPLIT
)


print()
print(
    "Training rows:",
    len(train_df)
)


print(
    "Testing rows :",
    len(test_df)
)


print(
    "Training Sample_IDs:",
    train_df["Sample_ID"].nunique()
)


print(
    "Testing Sample_IDs :",
    test_df["Sample_ID"].nunique()
)


# ======================================================================
# CSV LEVEL LEAKAGE CHECK
# ======================================================================

train_groups = set(
    train_df["Sample_ID"]
)


test_groups = set(
    test_df["Sample_ID"]
)


group_overlap = (
    train_groups.intersection(
        test_groups
    )
)


print()
print("=" * 70)
print("CSV SOURCE SAMPLE LEAKAGE CHECK")
print("=" * 70)


print(
    "Training Sample_IDs:",
    len(train_groups)
)


print(
    "Testing Sample_IDs :",
    len(test_groups)
)


print(
    "Overlapping Sample_IDs:",
    len(group_overlap)
)


if group_overlap:

    print()
    print(
        "OVERLAPPING SAMPLE IDs:"
    )

    for sample_id in sorted(
        group_overlap
    ):

        print(
            "-",
            sample_id
        )


    raise ValueError(
        "Sample_ID leakage detected."
    )


print()
print(
    "Leakage check: PASS"
)


# ======================================================================
# HDR DISCOVERY
# ======================================================================

print()
print("=" * 70)
print("DISCOVERING PAPAYA HDR FILES")
print("=" * 70)


hdr_files = glob.glob(
    os.path.join(
        PAPAYA_RAW_DIR,
        "**",
        "*.hdr"
    ),
    recursive=True
)


print()
print(
    "HDR files found:",
    len(hdr_files)
)


if not hdr_files:

    raise FileNotFoundError(
        "No .hdr files were found."
    )


# ======================================================================
# CREATE HDR LOOKUP
# ======================================================================

hdr_lookup = {}


for path in hdr_files:

    filename = os.path.basename(
        path
    ).lower()

    hdr_lookup[
        filename
    ] = path


print(
    "HDR lookup created."
)


# ======================================================================
# HELPER: FIND HDR PATH
# ======================================================================

def find_hdr_path(
    filename
):

    if pd.isna(
        filename
    ):

        return None


    filename = os.path.basename(
        str(filename)
    ).lower()


    return hdr_lookup.get(
        filename
    )


# ======================================================================
# HELPER: LOAD ENVI HDR
# ======================================================================

def load_envi_hdr(
    hdr_path
):
    """
    Load an ENVI hyperspectral file.

    Supports:
        BIP
        BIL
        BSQ

    Returns:
        cube with shape:
        (lines, samples, bands)
    """

    # ------------------------------------------------------------------
    # Read header
    # ------------------------------------------------------------------

    with open(
        hdr_path,
        "r",
        errors="ignore"
    ) as f:

        header_text = f.read()


    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def get_value(
        key,
        default=None
    ):

        pattern = (
            rf"(?im)^\s*"
            rf"{re.escape(key)}"
            rf"\s*=\s*(.+?)\s*$"
        )


        match = re.search(
            pattern,
            header_text
        )


        if not match:

            return default


        return match.group(
            1
        ).strip()


    samples = int(
        get_value(
            "samples"
        )
    )


    lines = int(
        get_value(
            "lines"
        )
    )


    bands = int(
        get_value(
            "bands"
        )
    )


    data_type = int(
        get_value(
            "data type"
        )
    )


    interleave = get_value(
        "interleave",
        "bsq"
    ).lower()


    byte_order = int(
        get_value(
            "byte order",
            "0"
        )
    )


    header_offset = int(
        get_value(
            "header offset",
            "0"
        )
    )


    # ------------------------------------------------------------------
    # ENVI data types
    # ------------------------------------------------------------------

    dtype_map = {

        1: np.uint8,

        2: np.int16,

        3: np.int32,

        4: np.float32,

        5: np.float64,

        12: np.uint16,

        13: np.uint32,

        14: np.int64,

        15: np.uint64
    }


    if data_type not in dtype_map:

        raise ValueError(
            f"Unsupported ENVI data type: "
            f"{data_type}"
        )


    dtype = np.dtype(
        dtype_map[data_type]
    )


    if byte_order == 1:

        dtype = dtype.newbyteorder(
            ">"
        )

    else:

        dtype = dtype.newbyteorder(
            "<"
        )


    # ------------------------------------------------------------------
    # Find binary data file
    # ------------------------------------------------------------------

    base_path = os.path.splitext(
        hdr_path
    )[0]


    candidates = [
        base_path,
        base_path + ".img"
    ]


    data_path = None


    for candidate in candidates:

        if os.path.exists(
            candidate
        ):

            data_path = candidate

            break


    if data_path is None:

        raise FileNotFoundError(
            f"Binary data file not found for:\n"
            f"{hdr_path}"
        )


    # ------------------------------------------------------------------
    # Read binary data
    # ------------------------------------------------------------------

    expected_values = (
        lines
        *
        samples
        *
        bands
    )


    data = np.fromfile(
        data_path,
        dtype=dtype,
        count=expected_values,
        offset=header_offset
    )


    if data.size != expected_values:

        raise ValueError(
            "\nUnexpected binary data size."
            f"\nExpected: {expected_values}"
            f"\nFound: {data.size}"
        )


    # ------------------------------------------------------------------
    # Convert to lines × samples × bands
    # ------------------------------------------------------------------

    if interleave == "bip":

        cube = data.reshape(
            lines,
            samples,
            bands
        )


    elif interleave == "bil":

        cube = data.reshape(
            lines,
            bands,
            samples
        )


        cube = np.transpose(
            cube,
            (
                0,
                2,
                1
            )
        )


    elif interleave == "bsq":

        cube = data.reshape(
            bands,
            lines,
            samples
        )


        cube = np.transpose(
            cube,
            (
                1,
                2,
                0
            )
        )


    else:

        raise ValueError(
            f"Unsupported ENVI interleave: "
            f"{interleave}"
        )


    return cube.astype(
        np.float32,
        copy=False
    )


# ======================================================================
# HELPER: PREPROCESS SPECTRA
# ======================================================================

def preprocess_spectra(
    spectra
):
    """
    Spectral preprocessing.

    1. Remove NaN/Inf rows
    2. Scale 4095-like values to 0-1
    3. Apply Savitzky-Golay smoothing
    """

    spectra = np.asarray(
        spectra,
        dtype=np.float32
    )


    # ------------------------------------------------------------------
    # Remove invalid rows
    # ------------------------------------------------------------------

    valid = np.isfinite(
        spectra
    ).all(
        axis=1
    )


    spectra = spectra[
        valid
    ]


    if len(
        spectra
    ) == 0:

        return spectra


    # ------------------------------------------------------------------
    # Detect 12-bit-like values
    # ------------------------------------------------------------------

    maximum = np.max(
        spectra
    )


    if maximum > 1.5:

        spectra = (
            spectra
            /
            4095.0
        )


    # ------------------------------------------------------------------
    # Savitzky-Golay smoothing
    # ------------------------------------------------------------------

    if (
        spectra.shape[1]
        >=
        SAVGOL_WINDOW
    ):

        spectra = savgol_filter(
            spectra,
            window_length=SAVGOL_WINDOW,
            polyorder=SAVGOL_POLYORDER,
            axis=1
        )


    return spectra.astype(
        np.float32
    )


# ======================================================================
# HELPER: EXTRACT PIXELS
# ======================================================================

def extract_pixels(
    cube,
    number_of_samples
):
    """
    Convert cube into:
        pixels × spectral bands

    Then randomly sample pixels.
    """

    if cube.ndim == 3:

        height = cube.shape[0]

        width = cube.shape[1]

        bands = cube.shape[2]


        spectra = cube.reshape(
            height * width,
            bands
        )


    elif cube.ndim == 2:

        spectra = cube


    else:

        raise ValueError(
            f"Unexpected cube shape: "
            f"{cube.shape}"
        )


    # ------------------------------------------------------------------
    # Remove invalid spectra
    # ------------------------------------------------------------------

    valid = np.isfinite(
        spectra
    ).all(
        axis=1
    )


    spectra = spectra[
        valid
    ]


    print(
        "Valid pixels:",
        len(spectra)
    )


    # ------------------------------------------------------------------
    # Random sampling
    # ------------------------------------------------------------------

    if len(
        spectra
    ) > number_of_samples:

        rng = np.random.default_rng(
            RANDOM_STATE
        )


        indices = rng.choice(
            len(spectra),
            size=number_of_samples,
            replace=False
        )


        spectra = spectra[
            indices
        ]


        print(
            "Sampled pixels:",
            len(spectra)
        )


    else:

        print(
            "Using all valid pixels:",
            len(spectra)
        )


    return spectra


# ======================================================================
# PROCESS TRAIN OR TEST DATA
# ======================================================================

def process_dataset(
    dataset_df,
    dataset_name
):

    print()
    print("=" * 70)

    print(
        f"PROCESSING {dataset_name.upper()} DATA"
    )

    print("=" * 70)


    all_spectra = []

    all_labels = []

    all_groups = []

    all_sources = []


    # ------------------------------------------------------------------
    # Unique HDR files
    # ------------------------------------------------------------------

    unique_hdrs = (
        dataset_df[
            "Subregions"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    print()

    print(
        "Unique HDR files:",
        len(unique_hdrs)
    )


    # ------------------------------------------------------------------
    # Process each HDR
    # ------------------------------------------------------------------

    for counter, hdr_name in enumerate(
        unique_hdrs,
        start=1
    ):

        print()

        print(
            "-" * 70
        )


        print(
            f"[{counter}/{len(unique_hdrs)}] "
            f"{hdr_name}"
        )


        # --------------------------------------------------------------
        # Find HDR
        # --------------------------------------------------------------

        hdr_path = find_hdr_path(
            hdr_name
        )


        if hdr_path is None:

            print(
                "WARNING: HDR file not found."
            )

            continue


        # --------------------------------------------------------------
        # Metadata rows
        # --------------------------------------------------------------

        rows = dataset_df[
            dataset_df[
                "Subregions"
            ]
            .astype(str)
            ==
            str(hdr_name)
        ]


        if len(
            rows
        ) == 0:

            print(
                "WARNING: No metadata rows."
            )

            continue


        # --------------------------------------------------------------
        # Label
        # --------------------------------------------------------------

        label_values = (
            rows[
                "Label_6Class"
            ]
            .unique()
        )


        # --------------------------------------------------------------
        # Sample ID
        # --------------------------------------------------------------

        sample_values = (
            rows[
                "Sample_ID"
            ]
            .unique()
        )


        if len(
            label_values
        ) != 1:

            raise ValueError(
                f"Multiple labels found for "
                f"{hdr_name}: "
                f"{label_values}"
            )


        if len(
            sample_values
        ) != 1:

            raise ValueError(
                f"Multiple Sample_IDs found for "
                f"{hdr_name}: "
                f"{sample_values}"
            )


        label = int(
            label_values[0]
        )


        sample_id = int(
            sample_values[0]
        )


        # --------------------------------------------------------------
        # Load cube
        # --------------------------------------------------------------

        cube = load_envi_hdr(
            hdr_path
        )


        print(
            "Cube shape:",
            cube.shape
        )


        print(
            "Cube dtype:",
            cube.dtype
        )


        print(
            "Label:",
            label
        )


        print(
            "Sample_ID:",
            sample_id
        )


        # --------------------------------------------------------------
        # Extract pixels
        # --------------------------------------------------------------

        spectra = extract_pixels(
            cube,
            SAMPLES_PER_CUBE
        )


        if len(
            spectra
        ) == 0:

            print(
                "WARNING: No valid pixels."
            )

            continue


        # --------------------------------------------------------------
        # Preprocess
        # --------------------------------------------------------------

        spectra = preprocess_spectra(
            spectra
        )


        if len(
            spectra
        ) == 0:

            print(
                "WARNING: No spectra after preprocessing."
            )

            continue


        print(
            "Processed spectra:",
            spectra.shape
        )


        # --------------------------------------------------------------
        # Store
        # --------------------------------------------------------------

        all_spectra.append(
            spectra
        )


        all_labels.extend(
            [label] * len(spectra)
        )


        all_groups.extend(
            [sample_id] * len(spectra)
        )


        all_sources.extend(
            [hdr_name] * len(spectra)
        )


    # ------------------------------------------------------------------
    # Verify data
    # ------------------------------------------------------------------

    if not all_spectra:

        raise RuntimeError(
            f"No usable spectra found for "
            f"{dataset_name}."
        )


    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    X = np.vstack(
        all_spectra
    ).astype(
        np.float32
    )


    y = np.asarray(
        all_labels
    )


    groups = np.asarray(
        all_groups
    )


    sources = np.asarray(
        all_sources
    )


    print()
    print(
        "=" * 70
    )


    print(
        f"{dataset_name.upper()} FINAL MATRIX"
    )


    print(
        "=" * 70
    )


    print(
        "Spectral matrix:",
        X.shape
    )


    print(
        "Labels:",
        y.shape
    )


    print(
        "Groups:",
        groups.shape
    )


    print(
        "Unique Sample_IDs:",
        len(
            np.unique(
                groups
            )
        )
    )


    return (
        X,
        y,
        groups,
        sources
    )


# ======================================================================
# PROCESS TRAINING DATA
# ======================================================================

(
    X_train_raw,
    y_train,
    train_hsi_groups,
    train_sources
) = process_dataset(
    train_df,
    "training"
)


# ======================================================================
# PROCESS TEST DATA
# ======================================================================

(
    X_test_raw,
    y_test,
    test_hsi_groups,
    test_sources
) = process_dataset(
    test_df,
    "testing"
)


# ======================================================================
# HSI LEVEL GROUP LEAKAGE CHECK
# ======================================================================

print()
print("=" * 70)
print("HSI SOURCE SAMPLE LEAKAGE CHECK")
print("=" * 70)


hsi_train_groups = set(
    train_hsi_groups
)


hsi_test_groups = set(
    test_hsi_groups
)


hsi_overlap = (
    hsi_train_groups.intersection(
        hsi_test_groups
    )
)


print(
    "Training HSI groups:",
    len(hsi_train_groups)
)


print(
    "Testing HSI groups :",
    len(hsi_test_groups)
)


print(
    "Overlap:",
    len(hsi_overlap)
)


if hsi_overlap:

    raise ValueError(
        "HSI Sample_ID leakage detected!"
    )


print(
    "HSI leakage check: PASS"
)


# ======================================================================
# SPECTRAL DIMENSION CHECK
# ======================================================================

print()
print("=" * 70)
print("SPECTRAL DIMENSION CHECK")
print("=" * 70)


print(
    "Training spectral bands:",
    X_train_raw.shape[1]
)


print(
    "Testing spectral bands :",
    X_test_raw.shape[1]
)


if (
    X_train_raw.shape[1]
    !=
    X_test_raw.shape[1]
):

    raise ValueError(
        "Training and testing have different "
        "numbers of spectral bands."
    )


# ======================================================================
# NUMERICAL VALIDATION
# ======================================================================

print()
print("=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)


train_nan = np.isnan(
    X_train_raw
).sum()


test_nan = np.isnan(
    X_test_raw
).sum()


train_inf = np.isinf(
    X_train_raw
).sum()


test_inf = np.isinf(
    X_test_raw
).sum()


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


# ======================================================================
# TRAINING-DERIVED MEAN CENTERING
# ======================================================================

print()
print("=" * 70)
print("TRAINING-DERIVED MEAN CENTERING")
print("=" * 70)


training_mean = (
    X_train_raw.mean(
        axis=0
    )
)


X_train_centered = (
    X_train_raw
    -
    training_mean
)


X_test_centered = (
    X_test_raw
    -
    training_mean
)


training_centering_error = np.mean(
    np.abs(
        X_train_centered.mean(
            axis=0
        )
    )
)


test_centered_mean = (
    X_test_centered.mean()
)


print(
    "Training mean absolute mean:",
    f"{training_centering_error:.10f}"
)


print(
    "Test mean after training centering:",
    f"{test_centered_mean:.10f}"
)


# ======================================================================
# PCA LIMIT
# ======================================================================

print()
print("=" * 70)
print("PCA LIMIT CHECK")
print("=" * 70)


training_samples = (
    X_train_centered.shape[0]
)


spectral_bands = (
    X_train_centered.shape[1]
)


maximum_valid_components = min(
    MAX_PCA_COMPONENTS,
    training_samples,
    spectral_bands
)


components_to_test = [
    value
    for value in PCA_COMPONENTS_TO_TEST
    if value <= maximum_valid_components
]


print(
    "Training spectra:",
    training_samples
)


print(
    "Original spectral bands:",
    spectral_bands
)


print(
    "Requested maximum PCs:",
    MAX_PCA_COMPONENTS
)


print(
    "Actual maximum valid PCs:",
    maximum_valid_components
)


print(
    "Components actually tested:",
    components_to_test
)


# ======================================================================
# PCA SENSITIVITY
# ======================================================================

print()
print("=" * 70)
print("STARTING PCA COMPONENT SENSITIVITY")
print("=" * 70)


results = []


for n_components in components_to_test:

    print()
    print(
        "-" * 70
    )


    print(
        f"EXPERIMENT: {n_components} PCA COMPONENTS"
    )


    print(
        "-" * 70
    )


    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    pca = PCA(
        n_components=n_components
    )


    print(
        "Fitting PCA using TRAINING data only..."
    )


    X_train_pca = (
        pca.fit_transform(
            X_train_centered
        )
    )


    print(
        "Transforming TEST data..."
    )


    X_test_pca = (
        pca.transform(
            X_test_centered
        )
    )


    # ------------------------------------------------------------------
    # Variance
    # ------------------------------------------------------------------

    variance_retained = (
        np.sum(
            pca.explained_variance_ratio_
        )
        *
        100
    )


    print(
        "Training PCA shape:",
        X_train_pca.shape
    )


    print(
        "Testing PCA shape :",
        X_test_pca.shape
    )


    print(
        f"Variance retained: "
        f"{variance_retained:.4f}%"
    )


    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------

    print(
        "Training Random Forest..."
    )


    model = RandomForestClassifier(
        n_estimators=RF_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    model.fit(
        X_train_pca,
        y_train
    )


    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    print(
        "Generating predictions..."
    )


    y_pred = model.predict(
        X_test_pca
    )


    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

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


    print()
    print(
        f"Accuracy        : "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Macro Precision : "
        f"{macro_precision * 100:.2f}%"
    )


    print(
        f"Macro Recall    : "
        f"{macro_recall * 100:.2f}%"
    )


    print(
        f"Macro F1        : "
        f"{macro_f1 * 100:.2f}%"
    )


    # ------------------------------------------------------------------
    # Store result
    # ------------------------------------------------------------------

    results.append({

        "PCA Components":
            n_components,

        "Variance Retained (%)":
            variance_retained,

        "Accuracy (%)":
            accuracy * 100,

        "Macro Precision (%)":
            macro_precision * 100,

        "Macro Recall (%)":
            macro_recall * 100,

        "Macro F1 (%)":
            macro_f1 * 100
    })


# ======================================================================
# RESULTS DATAFRAME
# ======================================================================

results_df = pd.DataFrame(
    results
)


# ======================================================================
# SAVE RESULTS
# ======================================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "papaya_extended_pca_sensitivity_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ======================================================================
# ACCURACY GRAPH
# ======================================================================

accuracy_graph = os.path.join(
    GRAPH_DIR,
    "papaya_extended_pca_component_accuracy.png"
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
    "Test Accuracy (%)"
)


plt.title(
    "Papaya HSI PCA Component Sensitivity"
)


plt.xticks(
    results_df[
        "PCA Components"
    ]
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


# ======================================================================
# MACRO F1 GRAPH
# ======================================================================

f1_graph = os.path.join(
    GRAPH_DIR,
    "papaya_extended_pca_component_macro_f1.png"
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    results_df[
        "PCA Components"
    ],
    results_df[
        "Macro F1 (%)"
    ],
    marker="o"
)


plt.xlabel(
    "Number of PCA Components"
)


plt.ylabel(
    "Macro F1 (%)"
)


plt.title(
    "Papaya HSI PCA Component vs Macro F1"
)


plt.xticks(
    results_df[
        "PCA Components"
    ]
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    f1_graph,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ======================================================================
# VARIANCE GRAPH
# ======================================================================

variance_graph = os.path.join(
    GRAPH_DIR,
    "papaya_extended_pca_component_variance.png"
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
    "Cumulative Variance Retained (%)"
)


plt.title(
    "Papaya HSI PCA Variance Retention"
)


plt.xticks(
    results_df[
        "PCA Components"
    ]
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


# ======================================================================
# BEST RESULT BY ACCURACY
# ======================================================================

best_accuracy_row = results_df.loc[
    results_df[
        "Accuracy (%)"
    ].idxmax()
]


# ======================================================================
# BEST RESULT BY MACRO F1
# ======================================================================

best_f1_row = results_df.loc[
    results_df[
        "Macro F1 (%)"
    ].idxmax()
]


# ======================================================================
# PRINT COMPLETE RESULTS
# ======================================================================

print()
print("=" * 70)
print("EXTENDED PCA SENSITIVITY RESULTS")
print("=" * 70)


print()


print(
    results_df.to_string(
        index=False
    )
)


# ======================================================================
# BEST ACCURACY
# ======================================================================

print()
print("=" * 70)
print("BEST RESULT BY TEST ACCURACY")
print("=" * 70)


print(
    "PCA components:",
    int(
        best_accuracy_row[
            "PCA Components"
        ]
    )
)


print(
    f"Variance retained: "
    f"{best_accuracy_row['Variance Retained (%)']:.4f}%"
)


print(
    f"Accuracy: "
    f"{best_accuracy_row['Accuracy (%)']:.2f}%"
)


print(
    f"Macro Precision: "
    f"{best_accuracy_row['Macro Precision (%)']:.2f}%"
)


print(
    f"Macro Recall: "
    f"{best_accuracy_row['Macro Recall (%)']:.2f}%"
)


print(
    f"Macro F1: "
    f"{best_accuracy_row['Macro F1 (%)']:.2f}%"
)


# ======================================================================
# BEST MACRO F1
# ======================================================================

print()
print("=" * 70)
print("BEST RESULT BY MACRO F1")
print("=" * 70)


print(
    "PCA components:",
    int(
        best_f1_row[
            "PCA Components"
        ]
    )
)


print(
    f"Variance retained: "
    f"{best_f1_row['Variance Retained (%)']:.4f}%"
)


print(
    f"Accuracy: "
    f"{best_f1_row['Accuracy (%)']:.2f}%"
)


print(
    f"Macro Precision: "
    f"{best_f1_row['Macro Precision (%)']:.2f}%"
)


print(
    f"Macro Recall: "
    f"{best_f1_row['Macro Recall (%)']:.2f}%"
)


print(
    f"Macro F1: "
    f"{best_f1_row['Macro F1 (%)']:.2f}%"
)


# ======================================================================
# FINAL VALIDATION SUMMARY
# ======================================================================

print()
print("=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)


print(
    "Training spectra:",
    len(X_train_raw)
)


print(
    "Testing spectra:",
    len(X_test_raw)
)


print(
    "Original spectral bands:",
    spectral_bands
)


print(
    "Training Sample_IDs:",
    len(train_groups)
)


print(
    "Testing Sample_IDs:",
    len(test_groups)
)


print(
    "Sample_ID overlap:",
    len(group_overlap)
)


print(
    "HSI Sample_ID overlap:",
    len(hsi_overlap)
)


print(
    "Training NaN:",
    train_nan
)


print(
    "Testing NaN:",
    test_nan
)


print(
    "Training Inf:",
    train_inf
)


print(
    "Testing Inf:",
    test_inf
)


# ======================================================================
# OUTPUT PATHS
# ======================================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)


print()
print(
    "Results:"
)


print(
    results_file
)


print()
print(
    "Accuracy graph:"
)


print(
    accuracy_graph
)


print()
print(
    "Macro F1 graph:"
)


print(
    f1_graph
)


print()
print(
    "Variance graph:"
)


print(
    variance_graph
)


# ======================================================================
# COMPLETED
# ======================================================================

print()
print("=" * 70)
print("PAPAYA EXTENDED PCA SENSITIVITY COMPLETED")
print("=" * 70)


print()
print(
    "Best PCA by accuracy:",
    int(
        best_accuracy_row[
            "PCA Components"
        ]
    )
)


print(
    f"Best accuracy: "
    f"{best_accuracy_row['Accuracy (%)']:.2f}%"
)


print(
    "Best PCA by Macro F1:",
    int(
        best_f1_row[
            "PCA Components"
        ]
    )
)


print(
    f"Best Macro F1: "
    f"{best_f1_row['Macro F1 (%)']:.2f}%"
)


print()
print(
    "PCA was fitted using training data only."
)


print(
    "Test data remained unseen during PCA fitting."
)


print(
    "Sample_ID leakage check: PASS"
)


print("=" * 70)