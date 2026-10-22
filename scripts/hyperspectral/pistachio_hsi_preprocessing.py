import os
import re
import gc

import numpy as np
import pandas as pd
import joblib

from sklearn.decomposition import PCA


# ======================================================================
# PISTACHIO HSI PREPROCESSING + PCA
# ======================================================================

print("=" * 70)
print("PISTACHIO HSI PREPROCESSING + PCA")
print("=" * 70)


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"


DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "pistachio_classification"
)


OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "pca"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TRAIN_FILE = os.path.join(
    DATA_DIR,
    "pistachio_train_group_split.csv"
)


TEST_FILE = os.path.join(
    DATA_DIR,
    "pistachio_test_group_split.csv"
)


# ======================================================================
# HSI CONFIGURATION
# ======================================================================

# Actual Pistachio ENVI dimensions:
#
# lines   = 256
# samples = 384
# bands   = 462
#
# BIL cube:
#
# (256, 462, 384)

EXPECTED_LINES = 256
EXPECTED_SAMPLES = 384
EXPECTED_BANDS = 462


# ======================================================================
# REFLECTANCE CONFIGURATION
# ======================================================================

REFLECTANCE_SCALE = 10000.0


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

RANDOM_STATE = 42


# ======================================================================
# SPECTRAL SAMPLING
# ======================================================================

# Each HSI cube contains approximately:
#
# 256 × 384 = 98,304 pixels
#
# We sample 2,000 valid foreground spectra
# from each cube.

SPECTRA_PER_TRAIN_CUBE = 2000
SPECTRA_PER_TEST_CUBE = 2000


# ======================================================================
# PCA CONFIGURATION
# ======================================================================

# IMPORTANT:
#
# We now examine ALL 462 possible components.
#
# This lets us determine exactly how many PCs
# are required for each variance target.

PCA_COMPONENTS_TO_TEST = EXPECTED_BANDS


# We will initially create the PCA dataset at
# the 95% variance point.
#
# We will NOT assume this is necessarily the
# best classifier configuration.
#
# After seeing the variance curve, we can compare
# other PCA sizes during classification.

INITIAL_VARIANCE_TARGET = 0.95


# ======================================================================
# ENVI HDR READER
# ======================================================================

def read_hdr(hdr_path):

    metadata = {}

    with open(
        hdr_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read()


    patterns = {

        "samples":
            r"(?im)^\s*samples\s*=\s*(\d+)",

        "lines":
            r"(?im)^\s*lines\s*=\s*(\d+)",

        "bands":
            r"(?im)^\s*bands\s*=\s*(\d+)",

        "data_type":
            r"(?im)^\s*data\s*type\s*=\s*(\d+)",

        "interleave":
            r"(?im)^\s*interleave\s*=\s*([^\s]+)",

        "byte_order":
            r"(?im)^\s*byte order\s*=\s*(\d+)"
    }


    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text
        )

        if match:

            value = match.group(1)


            if key in [
                "samples",
                "lines",
                "bands",
                "data_type",
                "byte_order"
            ]:

                value = int(value)

            else:

                value = value.lower()


            metadata[key] = value


    return metadata


# ======================================================================
# LOAD BIL CUBE
# ======================================================================

def load_bil_cube(
    bil_path,
    hdr_path
):

    metadata = read_hdr(
        hdr_path
    )


    required_fields = [

        "lines",
        "samples",
        "bands",
        "data_type",
        "interleave",
        "byte_order"
    ]


    for field in required_fields:

        if field not in metadata:

            raise ValueError(
                f"Missing '{field}' in HDR:\n"
                f"{hdr_path}"
            )


    lines = metadata["lines"]
    samples = metadata["samples"]
    bands = metadata["bands"]

    data_type = metadata["data_type"]
    interleave = metadata["interleave"]
    byte_order = metadata["byte_order"]


    # --------------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------------

    if lines != EXPECTED_LINES:

        raise ValueError(
            f"Unexpected lines in:\n{hdr_path}\n"
            f"Found: {lines}\n"
            f"Expected: {EXPECTED_LINES}"
        )


    if samples != EXPECTED_SAMPLES:

        raise ValueError(
            f"Unexpected samples in:\n{hdr_path}\n"
            f"Found: {samples}\n"
            f"Expected: {EXPECTED_SAMPLES}"
        )


    if bands != EXPECTED_BANDS:

        raise ValueError(
            f"Unexpected bands in:\n{hdr_path}\n"
            f"Found: {bands}\n"
            f"Expected: {EXPECTED_BANDS}"
        )


    # --------------------------------------------------------------
    # Validate BIL format
    # --------------------------------------------------------------

    if interleave != "bil":

        raise ValueError(
            f"Expected BIL interleave.\n"
            f"Found: {interleave}\n"
            f"File: {hdr_path}"
        )


    # --------------------------------------------------------------
    # ENVI data type
    #
    # 12 = uint16
    # --------------------------------------------------------------

    if data_type != 12:

        raise ValueError(
            f"Expected ENVI data type 12 (uint16).\n"
            f"Found: {data_type}\n"
            f"File: {hdr_path}"
        )


    # --------------------------------------------------------------
    # Byte order
    #
    # 0 = little endian
    # 1 = big endian
    # --------------------------------------------------------------

    if byte_order == 0:

        dtype = np.dtype("<u2")

    elif byte_order == 1:

        dtype = np.dtype(">u2")

    else:

        raise ValueError(
            f"Unsupported byte order: {byte_order}"
        )


    # --------------------------------------------------------------
    # Validate binary file size
    # --------------------------------------------------------------

    expected_values = (
        lines
        * bands
        * samples
    )


    expected_bytes = (
        expected_values
        * dtype.itemsize
    )


    actual_bytes = os.path.getsize(
        bil_path
    )


    if actual_bytes != expected_bytes:

        raise ValueError(
            f"BIL file size mismatch:\n"
            f"{bil_path}\n\n"
            f"Expected bytes: {expected_bytes}\n"
            f"Actual bytes:   {actual_bytes}"
        )


    # --------------------------------------------------------------
    # Load binary data
    # --------------------------------------------------------------

    data = np.fromfile(
        bil_path,
        dtype=dtype
    )


    # --------------------------------------------------------------
    # BIL layout
    #
    # lines × bands × samples
    # --------------------------------------------------------------

    cube = data.reshape(
        (
            lines,
            bands,
            samples
        )
    )


    return cube


# ======================================================================
# REFLECTANCE CONVERSION
# ======================================================================

def convert_to_reflectance(
    cube
):

    reflectance = (
        cube.astype(
            np.float32
        )
        /
        REFLECTANCE_SCALE
    )


    # Keep physically reasonable range.
    #
    # This prevents extreme invalid values
    # from entering subsequent processing.

    reflectance = np.clip(
        reflectance,
        0.0,
        1.5
    )


    return reflectance


# ======================================================================
# FOREGROUND SPECTRUM EXTRACTION
# ======================================================================

def extract_spectra(
    cube,
    number_of_spectra,
    rng
):

    lines, bands, samples = cube.shape


    # --------------------------------------------------------------
    # Mean reflectance for every spatial pixel
    # --------------------------------------------------------------

    mean_reflectance = np.mean(
        cube,
        axis=1
    )


    # --------------------------------------------------------------
    # Background mask
    #
    # Very dark pixels are treated as background.
    # --------------------------------------------------------------

    valid_mask = (
        mean_reflectance
        > 0.02
    )


    valid_positions = np.argwhere(
        valid_mask
    )


    if len(valid_positions) == 0:

        raise ValueError(
            "No valid foreground pixels found."
        )


    # --------------------------------------------------------------
    # Number of spectra
    # --------------------------------------------------------------

    number_to_take = min(
        number_of_spectra,
        len(valid_positions)
    )


    # --------------------------------------------------------------
    # Reproducible random sampling
    # --------------------------------------------------------------

    selected_indices = rng.choice(
        len(valid_positions),
        size=number_to_take,
        replace=False
    )


    selected_positions = (
        valid_positions[
            selected_indices
        ]
    )


    # --------------------------------------------------------------
    # Allocate spectral matrix
    #
    # rows    = sampled pixels
    # columns = spectral bands
    # --------------------------------------------------------------

    spectra = np.empty(
        (
            number_to_take,
            bands
        ),
        dtype=np.float32
    )


    # --------------------------------------------------------------
    # Extract spectra
    # --------------------------------------------------------------

    for i, (
        row_index,
        column_index
    ) in enumerate(
        selected_positions
    ):

        spectra[i] = (
            cube[
                row_index,
                :,
                column_index
            ]
        )


    return spectra


# ======================================================================
# STANDARD NORMAL VARIATE
# ======================================================================

def normalize_spectra(
    spectra
):

    # --------------------------------------------------------------
    # SNV:
    #
    # normalized = (spectrum - mean) / standard deviation
    #
    # Each spectrum is normalized independently.
    # --------------------------------------------------------------

    means = np.mean(
        spectra,
        axis=1,
        keepdims=True
    )


    stds = np.std(
        spectra,
        axis=1,
        keepdims=True
    )


    stds[
        stds < 1e-8
    ] = 1.0


    normalized = (
        spectra
        - means
    ) / stds


    return normalized.astype(
        np.float32
    )


# ======================================================================
# LOAD TRAIN / TEST MANIFESTS
# ======================================================================

print(
    "\nLoading train/test manifests..."
)


if not os.path.exists(
    TRAIN_FILE
):

    raise FileNotFoundError(
        f"Training manifest not found:\n"
        f"{TRAIN_FILE}"
    )


if not os.path.exists(
    TEST_FILE
):

    raise FileNotFoundError(
        f"Testing manifest not found:\n"
        f"{TEST_FILE}"
    )


train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print(
    "\nTraining cubes:",
    len(train_df)
)

print(
    "Testing cubes :",
    len(test_df)
)


# ======================================================================
# REQUIRED COLUMNS
# ======================================================================

required_columns = [

    "Sample_ID",
    "Level",
    "AFB1_ug_kg",
    "Binary_Label",
    "Binary_Class",
    "BIL_Path",
    "HDR_Path",
    "Group_ID"
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


print(
    "\nManifest columns: PASS"
)


# ======================================================================
# SOURCE-LEVEL LEAKAGE CHECK
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "SOURCE-LEVEL LEAKAGE CHECK"
)

print(
    "=" * 70
)


train_groups = set(
    train_df["Group_ID"]
)

test_groups = set(
    test_df["Group_ID"]
)


group_overlap = (
    train_groups
    &
    test_groups
)


print(
    "\nTraining levels:",
    len(train_groups)
)

print(
    "Testing levels :",
    len(test_groups)
)

print(
    "Overlapping levels:",
    len(group_overlap)
)


if group_overlap:

    raise ValueError(
        f"Contamination-level leakage detected:\n"
        f"{sorted(group_overlap)}"
    )


print(
    "\nPASS: No contamination-level overlap."
)


# ======================================================================
# RANDOM NUMBER GENERATOR
# ======================================================================

rng = np.random.default_rng(
    RANDOM_STATE
)


# ======================================================================
# PROCESS TRAINING CUBES
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "PROCESSING TRAINING CUBES"
)

print(
    "=" * 70
)


training_spectra = []
training_metadata = []


for index, row in train_df.iterrows():

    print(
        "\n" + "-" * 70
    )


    print(
        f"[{index + 1}/{len(train_df)}]"
    )


    sample_id = row[
        "Sample_ID"
    ]


    level = int(
        row["Level"]
    )


    afb1 = float(
        row["AFB1_ug_kg"]
    )


    binary_class = row[
        "Binary_Class"
    ]


    bil_path = row[
        "BIL_Path"
    ]


    hdr_path = row[
        "HDR_Path"
    ]


    print(
        "Sample:",
        sample_id
    )


    print(
        "Level:",
        level
    )


    print(
        "AFB1:",
        afb1,
        "µg/kg"
    )


    print(
        "Class:",
        binary_class
    )


    # --------------------------------------------------------------
    # Validate files
    # --------------------------------------------------------------

    if not os.path.exists(
        bil_path
    ):

        raise FileNotFoundError(
            f"BIL file not found:\n{bil_path}"
        )


    if not os.path.exists(
        hdr_path
    ):

        raise FileNotFoundError(
            f"HDR file not found:\n{hdr_path}"
        )


    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    raw_cube = load_bil_cube(
        bil_path,
        hdr_path
    )


    print(
        "Raw cube:",
        raw_cube.shape
    )


    # --------------------------------------------------------------
    # Reflectance
    # --------------------------------------------------------------

    reflectance = (
        convert_to_reflectance(
            raw_cube
        )
    )


    del raw_cube


    print(
        "Reflectance conversion: PASS"
    )


    # --------------------------------------------------------------
    # Extract spectra
    # --------------------------------------------------------------

    spectra = extract_spectra(
        reflectance,
        SPECTRA_PER_TRAIN_CUBE,
        rng
    )


    del reflectance


    print(
        "Spectra extracted:",
        spectra.shape
    )


    # --------------------------------------------------------------
    # SNV
    # --------------------------------------------------------------

    spectra = normalize_spectra(
        spectra
    )


    print(
        "SNV normalization: PASS"
    )


    # --------------------------------------------------------------
    # Save spectra
    # --------------------------------------------------------------

    training_spectra.append(
        spectra
    )


    metadata_record = {

        "Sample_ID":
            sample_id,

        "Level":
            level,

        "AFB1_ug_kg":
            afb1,

        "Binary_Label":
            int(
                row["Binary_Label"]
            ),

        "Binary_Class":
            binary_class,

        "Group_ID":
            int(
                row["Group_ID"]
            )
    }


    for _ in range(
        len(spectra)
    ):

        training_metadata.append(
            metadata_record.copy()
        )


    del spectra

    gc.collect()


# ======================================================================
# COMBINE TRAINING SPECTRA
# ======================================================================

X_train_spectral = np.vstack(
    training_spectra
)


del training_spectra


train_metadata_df = pd.DataFrame(
    training_metadata
)


del training_metadata


print(
    "\n" + "=" * 70
)

print(
    "TRAINING SPECTRA CREATED"
)

print(
    "=" * 70
)


print(
    "\nTraining spectral matrix:",
    X_train_spectral.shape
)


print(
    "Training metadata rows:",
    len(train_metadata_df)
)


# ======================================================================
# TRAINING NUMERICAL VALIDATION
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING SPECTRAL VALIDATION"
)

print(
    "=" * 70
)


train_nan = np.isnan(
    X_train_spectral
).sum()


train_inf = np.isinf(
    X_train_spectral
).sum()


print(
    "\nNaN:",
    train_nan
)

print(
    "Inf:",
    train_inf
)


if (
    train_nan > 0
    or
    train_inf > 0
):

    raise ValueError(
        "NaN or Inf detected in training spectra."
    )


print(
    "\nTraining spectral validation: PASS"
)


# ======================================================================
# FIT FULL PCA
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "FITTING FULL PCA"
)

print(
    "=" * 70
)


print(
    "\nOriginal spectral bands:",
    X_train_spectral.shape[1]
)


print(
    "Maximum PCs examined:",
    PCA_COMPONENTS_TO_TEST
)


print(
    "\nPCA is fitted on TRAINING spectra ONLY."
)


print(
    "This prevents test-data leakage."
)


# --------------------------------------------------------------
# Full PCA
#
# We use all 462 components so that the complete variance
# curve can be examined.
#
# svd_solver='full' is appropriate here because:
#
# 84,000 samples × 462 bands
#
# is manageable and we want the complete variance spectrum.
# --------------------------------------------------------------

pca = PCA(
    n_components=PCA_COMPONENTS_TO_TEST,
    svd_solver="full"
)


print(
    "\nStarting PCA fitting..."
)


X_train_pca_full = pca.fit_transform(
    X_train_spectral
)


print(
    "PCA fitting completed."
)


# ======================================================================
# EXPLAINED VARIANCE
# ======================================================================

explained_variance = (
    pca.explained_variance_ratio_
)


cumulative_variance = np.cumsum(
    explained_variance
)


print(
    "\n" + "=" * 70
)

print(
    "PCA VARIANCE ANALYSIS"
)

print(
    "=" * 70
)


print(
    "\nVariance retention:"
)


variance_targets = [

    0.90,
    0.95,
    0.99,
    0.995,
    0.999
]


variance_results = []


for target in variance_targets:

    components_needed = (
        np.searchsorted(
            cumulative_variance,
            target
        )
        + 1
    )


    if components_needed > EXPECTED_BANDS:

        print(
            f"  {target * 100:.1f}% : "
            f"> {EXPECTED_BANDS} PCs"
        )


        variance_results.append({

            "target_variance":
                target,

            "components_required":
                np.nan,

            "actual_variance":
                cumulative_variance[-1]
        })


    else:

        actual_variance = (
            cumulative_variance[
                components_needed - 1
            ]
        )


        print(
            f"  {target * 100:.1f}% : "
            f"{components_needed} PCs "
            f"("
            f"{actual_variance * 100:.4f}%"
            f")"
        )


        variance_results.append({

            "target_variance":
                target,

            "components_required":
                components_needed,

            "actual_variance":
                actual_variance
        })


# ======================================================================
# PRINT SELECTED PCA INFORMATION
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "PCA SUMMARY"
)

print(
    "=" * 70
)


print(
    "\nFirst 10 PCs cumulative variance:"
)


for i in range(
    min(10, len(cumulative_variance))
):

    print(
        f"PC{i + 1:03d}: "
        f"{cumulative_variance[i] * 100:.4f}%"
    )


print(
    "\nSelected checkpoints:"
)


for pc_number in [
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    462
]:

    if pc_number <= len(
        cumulative_variance
    ):

        print(
            f"PC{pc_number:03d}: "
            f"{cumulative_variance[pc_number - 1] * 100:.4f}%"
        )


# ======================================================================
# SAVE COMPLETE PCA VARIANCE TABLE
# ======================================================================

variance_df = pd.DataFrame({

    "PC":
        np.arange(
            1,
            len(explained_variance) + 1
        ),

    "explained_variance_ratio":
        explained_variance,

    "cumulative_variance":
        cumulative_variance
})


variance_table_path = os.path.join(
    OUTPUT_DIR,
    "pistachio_pca_full_variance.csv"
)


variance_df.to_csv(
    variance_table_path,
    index=False
)


print(
    "\nFull PCA variance table saved:"
)


print(
    variance_table_path
)


# ======================================================================
# SAVE PCA MODEL
# ======================================================================

full_pca_model_path = os.path.join(
    OUTPUT_DIR,
    "pistachio_pca_full_model.joblib"
)


joblib.dump(
    pca,
    full_pca_model_path
)


print(
    "\nFull PCA model saved:"
)


print(
    full_pca_model_path
)


# ======================================================================
# DETERMINE INITIAL PCA SIZE
# ======================================================================

initial_components = (
    np.searchsorted(
        cumulative_variance,
        INITIAL_VARIANCE_TARGET
    )
    + 1
)


initial_actual_variance = (
    cumulative_variance[
        initial_components - 1
    ]
)


print(
    "\n" + "=" * 70
)

print(
    "INITIAL PCA DATASET CONFIGURATION"
)

print(
    "=" * 70
)


print(
    "\nInitial variance target:",
    f"{INITIAL_VARIANCE_TARGET * 100:.1f}%"
)


print(
    "Initial selected PCs:",
    initial_components
)


print(
    "Actual variance retained:",
    f"{initial_actual_variance * 100:.4f}%"
)


# ======================================================================
# CREATE TRAINING PCA DATA
# ======================================================================

X_train_pca = (
    X_train_pca_full[
        :,
        :initial_components
    ]
)


# ======================================================================
# PROCESS TESTING CUBES
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "PROCESSING TESTING CUBES"
)

print(
    "=" * 70
)


testing_spectra = []
testing_metadata = []


for index, row in test_df.iterrows():

    print(
        "\n" + "-" * 70
    )


    print(
        f"[{index + 1}/{len(test_df)}]"
    )


    sample_id = row[
        "Sample_ID"
    ]


    level = int(
        row["Level"]
    )


    afb1 = float(
        row["AFB1_ug_kg"]
    )


    binary_class = row[
        "Binary_Class"
    ]


    bil_path = row[
        "BIL_Path"
    ]


    hdr_path = row[
        "HDR_Path"
    ]


    print(
        "Sample:",
        sample_id
    )


    print(
        "Level:",
        level
    )


    print(
        "AFB1:",
        afb1,
        "µg/kg"
    )


    print(
        "Class:",
        binary_class
    )


    # --------------------------------------------------------------
    # Validate files
    # --------------------------------------------------------------

    if not os.path.exists(
        bil_path
    ):

        raise FileNotFoundError(
            f"BIL file not found:\n{bil_path}"
        )


    if not os.path.exists(
        hdr_path
    ):

        raise FileNotFoundError(
            f"HDR file not found:\n{hdr_path}"
        )


    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    raw_cube = load_bil_cube(
        bil_path,
        hdr_path
    )


    print(
        "Raw cube:",
        raw_cube.shape
    )


    # --------------------------------------------------------------
    # Reflectance
    # --------------------------------------------------------------

    reflectance = (
        convert_to_reflectance(
            raw_cube
        )
    )


    del raw_cube


    print(
        "Reflectance conversion: PASS"
    )


    # --------------------------------------------------------------
    # Extract spectra
    # --------------------------------------------------------------

    spectra = extract_spectra(
        reflectance,
        SPECTRA_PER_TEST_CUBE,
        rng
    )


    del reflectance


    print(
        "Spectra extracted:",
        spectra.shape
    )


    # --------------------------------------------------------------
    # SNV
    # --------------------------------------------------------------

    spectra = normalize_spectra(
        spectra
    )


    print(
        "SNV normalization: PASS"
    )


    testing_spectra.append(
        spectra
    )


    metadata_record = {

        "Sample_ID":
            sample_id,

        "Level":
            level,

        "AFB1_ug_kg":
            afb1,

        "Binary_Label":
            int(
                row["Binary_Label"]
            ),

        "Binary_Class":
            binary_class,

        "Group_ID":
            int(
                row["Group_ID"]
            )
    }


    for _ in range(
        len(spectra)
    ):

        testing_metadata.append(
            metadata_record.copy()
        )


    del spectra

    gc.collect()


# ======================================================================
# COMBINE TESTING SPECTRA
# ======================================================================

X_test_spectral = np.vstack(
    testing_spectra
)


del testing_spectra


test_metadata_df = pd.DataFrame(
    testing_metadata
)


del testing_metadata


print(
    "\n" + "=" * 70
)

print(
    "TESTING SPECTRA CREATED"
)

print(
    "=" * 70
)


print(
    "\nTesting spectral matrix:",
    X_test_spectral.shape
)


print(
    "Testing metadata rows:",
    len(test_metadata_df)
)


# ======================================================================
# TESTING NUMERICAL VALIDATION
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "TESTING SPECTRAL VALIDATION"
)

print(
    "=" * 70
)


test_nan = np.isnan(
    X_test_spectral
).sum()


test_inf = np.isinf(
    X_test_spectral
).sum()


print(
    "\nNaN:",
    test_nan
)


print(
    "Inf:",
    test_inf
)


if (
    test_nan > 0
    or
    test_inf > 0
):

    raise ValueError(
        "NaN or Inf detected in testing spectra."
    )


print(
    "\nTesting spectral validation: PASS"
)


# ======================================================================
# TRANSFORM TESTING DATA
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "TRANSFORMING TESTING DATA"
)

print(
    "=" * 70
)


print(
    "\nApplying TRAINING-FITTED PCA..."
)


X_test_pca_full = pca.transform(
    X_test_spectral
)


X_test_pca = (
    X_test_pca_full[
        :,
        :initial_components
    ]
)


del X_test_pca_full
del X_test_spectral


print(
    "\nTesting PCA matrix:",
    X_test_pca.shape
)


# ======================================================================
# PCA NUMERICAL VALIDATION
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "PCA NUMERICAL VALIDATION"
)

print(
    "=" * 70
)


train_pca_nan = np.isnan(
    X_train_pca
).sum()


test_pca_nan = np.isnan(
    X_test_pca
).sum()


train_pca_inf = np.isinf(
    X_train_pca
).sum()


test_pca_inf = np.isinf(
    X_test_pca
).sum()


print(
    "\nTraining NaN:",
    train_pca_nan
)


print(
    "Testing NaN :",
    test_pca_nan
)


print(
    "Training Inf:",
    train_pca_inf
)


print(
    "Testing Inf :",
    test_pca_inf
)


if (
    train_pca_nan > 0
    or
    test_pca_nan > 0
    or
    train_pca_inf > 0
    or
    test_pca_inf > 0
):

    raise ValueError(
        "NaN or Inf detected in PCA data."
    )


print(
    "\nPCA numerical validation: PASS"
)


# ======================================================================
# CREATE PCA DATAFRAMES
# ======================================================================

pc_columns = [

    f"PC{i + 1}"

    for i in range(
        initial_components
    )
]


train_features_df = pd.DataFrame(
    X_train_pca,
    columns=pc_columns
)


test_features_df = pd.DataFrame(
    X_test_pca,
    columns=pc_columns
)


train_pca_df = pd.concat(
    [

        train_metadata_df.reset_index(
            drop=True
        ),

        train_features_df.reset_index(
            drop=True
        )

    ],
    axis=1
)


test_pca_df = pd.concat(
    [

        test_metadata_df.reset_index(
            drop=True
        ),

        test_features_df.reset_index(
            drop=True
        )

    ],
    axis=1
)


# ======================================================================
# CREATE OUTPUT DIRECTORY
# ======================================================================

pca_dir = os.path.join(
    OUTPUT_DIR,
    f"pca_{initial_components}pc"
)


os.makedirs(
    pca_dir,
    exist_ok=True
)


# ======================================================================
# OUTPUT FILES
# ======================================================================

train_output = os.path.join(
    pca_dir,
    f"pistachio_train_{initial_components}pc.csv"
)


test_output = os.path.join(
    pca_dir,
    f"pistachio_test_{initial_components}pc.csv"
)


pca_model_output = os.path.join(
    pca_dir,
    "pistachio_pca_model.joblib"
)


# ======================================================================
# SAVE PCA DATA
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "SAVING PCA DATA"
)

print(
    "=" * 70
)


train_pca_df.to_csv(
    train_output,
    index=False
)


print(
    "\nTraining PCA file:"
)

print(
    train_output
)


test_pca_df.to_csv(
    test_output,
    index=False
)


print(
    "\nTesting PCA file:"
)

print(
    test_output
)


# ======================================================================
# SAVE INITIAL PCA MODEL
# ======================================================================

joblib.dump(
    pca,
    pca_model_output
)


print(
    "\nPCA model:"
)

print(
    pca_model_output
)


# ======================================================================
# FINAL SAMPLE ID LEAKAGE CHECK
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL SAMPLE_ID LEAKAGE CHECK"
)

print(
    "=" * 70
)


train_sample_ids = set(
    train_pca_df[
        "Sample_ID"
    ]
)


test_sample_ids = set(
    test_pca_df[
        "Sample_ID"
    ]
)


sample_overlap = (
    train_sample_ids
    &
    test_sample_ids
)


print(
    "\nTraining Sample_IDs:",
    len(train_sample_ids)
)


print(
    "Testing Sample_IDs :",
    len(test_sample_ids)
)


print(
    "Sample_ID overlap:",
    len(sample_overlap)
)


if sample_overlap:

    raise ValueError(
        f"Sample_ID leakage detected:\n"
        f"{sorted(sample_overlap)}"
    )


print(
    "\nPASS: No Sample_ID overlap."
)


# ======================================================================
# FINAL GROUP LEAKAGE CHECK
# ======================================================================

final_train_groups = set(
    train_pca_df[
        "Group_ID"
    ]
)


final_test_groups = set(
    test_pca_df[
        "Group_ID"
    ]
)


final_group_overlap = (
    final_train_groups
    &
    final_test_groups
)


print(
    "\n" + "=" * 70
)

print(
    "FINAL GROUP LEAKAGE CHECK"
)

print(
    "=" * 70
)


print(
    "\nTraining groups:",
    len(final_train_groups)
)


print(
    "Testing groups :",
    len(final_test_groups)
)


print(
    "Group overlap:",
    len(final_group_overlap)
)


if final_group_overlap:

    raise ValueError(
        f"Contamination-level leakage detected:\n"
        f"{sorted(final_group_overlap)}"
    )


print(
    "\nPASS: No contamination-level leakage."
)


# ======================================================================
# FINAL CLASS DISTRIBUTION
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL PCA CLASS DISTRIBUTION"
)

print(
    "=" * 70
)


print(
    "\nTraining:"
)


print(
    train_pca_df[
        "Binary_Class"
    ].value_counts()
)


print(
    "\nTesting:"
)


print(
    test_pca_df[
        "Binary_Class"
    ].value_counts()
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print(
    "\n" + "=" * 70
)

print(
    "PISTACHIO HSI PREPROCESSING COMPLETED"
)

print(
    "=" * 70
)


print(
f"""
Original spectral bands : {EXPECTED_BANDS}

Spatial dimensions      : {EXPECTED_LINES} x {EXPECTED_SAMPLES}

HSI cube shape          : ({EXPECTED_LINES}, {EXPECTED_BANDS}, {EXPECTED_SAMPLES})

Reflectance scale       : {REFLECTANCE_SCALE}

Normalization           : SNV

Training cubes          : {len(train_df)}

Testing cubes           : {len(test_df)}

Training levels         : {len(train_groups)}

Testing levels          : {len(test_groups)}

Training spectra        : {len(train_pca_df)}

Testing spectra         : {len(test_pca_df)}

Spectra per train cube  : {SPECTRA_PER_TRAIN_CUBE}

Spectra per test cube   : {SPECTRA_PER_TEST_CUBE}

Full PCA components     : {EXPECTED_BANDS}

Initial variance target : {INITIAL_VARIANCE_TARGET * 100:.1f}%

Initial selected PCs    : {initial_components}

Variance retained       : {initial_actual_variance * 100:.4f}%

Sample_ID overlap       : {len(sample_overlap)}

Group overlap           : {len(final_group_overlap)}

Source-level leakage    : PASS


FULL PCA VARIANCE TABLE:

{variance_table_path}


FULL PCA MODEL:

{full_pca_model_path}


TRAINING PCA DATA:

{train_output}


TESTING PCA DATA:

{test_output}


INITIAL PCA MODEL:

{pca_model_output}
"""
)


print(
"=" * 70
)


print(
"""
IMPORTANT:

The initial PCA dataset uses the 95% variance point.

This is NOT automatically the final classifier PCA size.

The complete 462-PC variance table has been saved.

Next we will examine the PCA variance results and
choose the most appropriate dimensionality for the
Random Forest, SVM, and XGBoost comparison.
"""
)


print(
"=" * 70
)