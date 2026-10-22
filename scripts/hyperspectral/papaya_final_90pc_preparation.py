# ==============================================================
# PAPAYA HSI FINAL 90-PC DATASET CREATION
# ==============================================================
#
# Purpose:
#   Create the final Papaya HSI classification dataset using
#   the best PCA configuration found during sensitivity testing.
#
# Selected PCA:
#   90 components
#
# Important:
#   - PCA is fitted ONLY on training spectra.
#   - Test spectra are transformed using the training PCA.
#   - Sample_ID groups are never split between train/test.
#   - No pixel/sample leakage.
#
# Input:
#   papaya_train_split.csv
#   papaya_test_split.csv
#
# Output:
#   papaya_train_90pc.csv
#   papaya_test_90pc.csv
#   papaya_90pc_dataset.csv
# ==============================================================

import os
import numpy as np
import pandas as pd
import spectral
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = r"D:\Projects\AI-Food-Quality-Platform"

RAW_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "hyperspectral",
    "papaya_hsi"
)

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "hyperspectral",
    "papaya_classification"
)

OUTPUT_DIR = os.path.join(
    SPLIT_DIR,
    "pca_90pc"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_SPLIT = os.path.join(
    SPLIT_DIR,
    "papaya_train_split.csv"
)

TEST_SPLIT = os.path.join(
    SPLIT_DIR,
    "papaya_test_split.csv"
)

# --------------------------------------------------------------
# PCA configuration
# --------------------------------------------------------------

N_COMPONENTS = 90

# Number of pixels to sample from each HSI cube
PIXELS_PER_CUBE = 200

# Savitzky-Golay smoothing
SG_WINDOW = 7
SG_POLYORDER = 2

RANDOM_STATE = 42

# ==============================================================
# DISPLAY
# ==============================================================

print("=" * 70)
print("PAPAYA HSI FINAL 90-PC DATASET CREATION")
print("=" * 70)

print()
print("Selected PCA components :", N_COMPONENTS)
print("Original spectral bands : 150")
print("Pixels sampled/cube     :", PIXELS_PER_CUBE)
print("Smoothing window        :", SG_WINDOW)
print("Polynomial order        :", SG_POLYORDER)

# ==============================================================
# LOAD TRAIN / TEST SPLITS
# ==============================================================

print()
print("=" * 70)
print("LOADING GROUP SPLITS")
print("=" * 70)

train_df = pd.read_csv(TRAIN_SPLIT)
test_df = pd.read_csv(TEST_SPLIT)

print()
print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))

required_columns = [
    "Subregions",
    "RGB_Filename",
    "Header_Cropped",
    "Sample_ID",
    "Label_6Class",
    "Subregions_Img"
]

for col in required_columns:
    if col not in train_df.columns:
        raise ValueError(
            f"Missing required training column: {col}"
        )

    if col not in test_df.columns:
        raise ValueError(
            f"Missing required testing column: {col}"
        )

print()
print("Required columns validated.")

# ==============================================================
# SOURCE HSI DIRECTORY
# ==============================================================

print()
print("=" * 70)
print("PAPAYA HSI DIRECTORY")
print("=" * 70)

print(RAW_DIR)

if not os.path.exists(RAW_DIR):
    raise FileNotFoundError(
        f"Papaya HSI directory not found:\n{RAW_DIR}"
    )

# ==============================================================
# HELPER: LOCATE HDR FILE
# ==============================================================

def find_hdr(filename):

    filename = str(filename)

    # Direct path
    direct_path = os.path.join(RAW_DIR, filename)

    if os.path.exists(direct_path):
        return direct_path

    # Search recursively
    for root, dirs, files in os.walk(RAW_DIR):

        for file in files:

            if file.lower() == filename.lower():

                return os.path.join(root, file)

    return None


# ==============================================================
# HELPER: PROCESS HSI CUBE
# ==============================================================

def process_cube(hdr_path, label, sample_id):

    print()
    print("-" * 70)
    print("Processing:", os.path.basename(hdr_path))
    print("Label     :", label)
    print("Sample_ID :", sample_id)

    # ----------------------------------------------------------
    # Open ENVI image
    # ----------------------------------------------------------

    img = spectral.open_image(hdr_path)

    cube = np.asarray(
        img.load()
    ).astype(np.float32)

    print("Cube shape:", cube.shape)
    print("Cube dtype:", cube.dtype)

    # ----------------------------------------------------------
    # Validate dimensions
    # ----------------------------------------------------------

    if cube.ndim != 3:

        raise ValueError(
            f"Expected 3D HSI cube, got shape {cube.shape}"
        )

    bands = cube.shape[2]

    print("Spectral bands:", bands)

    if bands != 150:

        raise ValueError(
            f"Expected 150 spectral bands, got {bands}"
        )

    # ----------------------------------------------------------
    # Convert to pixel x spectral matrix
    # ----------------------------------------------------------

    spectra = cube.reshape(
        -1,
        bands
    )

    # ----------------------------------------------------------
    # Remove invalid pixels
    # ----------------------------------------------------------

    valid_mask = np.isfinite(
        spectra
    ).all(axis=1)

    spectra = spectra[
        valid_mask
    ]

    # Remove completely zero pixels
    nonzero_mask = np.any(
        spectra != 0,
        axis=1
    )

    spectra = spectra[
        nonzero_mask
    ]

    print("Valid pixels:", len(spectra))

    # ----------------------------------------------------------
    # Sample pixels
    # ----------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_count = min(
        PIXELS_PER_CUBE,
        len(spectra)
    )

    if sample_count == 0:

        raise ValueError(
            "No valid pixels found."
        )

    indices = rng.choice(
        len(spectra),
        size=sample_count,
        replace=False
    )

    spectra = spectra[
        indices
    ]

    print("Sampled pixels:", len(spectra))

    # ----------------------------------------------------------
    # Detect scaling
    #
    # Some HSI data may be normalized 0-1 while others may
    # use 12-bit-like values around 4095.
    # ----------------------------------------------------------

    maximum = np.max(spectra)

    if maximum > 100:

        print(
            "Scale: high-range -> dividing by 4095"
        )

        spectra = spectra / 4095.0

    else:

        print(
            "Scale: 0-1-like -> unchanged"
        )

    # ----------------------------------------------------------
    # Savitzky-Golay smoothing
    # ----------------------------------------------------------

    print(
        f"Smoothing: window={SG_WINDOW}, "
        f"polyorder={SG_POLYORDER}"
    )

    spectra = savgol_filter(
        spectra,
        window_length=SG_WINDOW,
        polyorder=SG_POLYORDER,
        axis=1
    )

    # ----------------------------------------------------------
    # Numerical validation
    # ----------------------------------------------------------

    if not np.isfinite(spectra).all():

        raise ValueError(
            "NaN or Inf detected after preprocessing."
        )

    print(
        "Processed spectra:",
        spectra.shape
    )

    return spectra


# ==============================================================
# PROCESS TRAINING CUBES
# ==============================================================

print()
print("=" * 70)
print("PROCESSING TRAINING CUBES")
print("=" * 70)

train_spectra_list = []
train_labels_list = []
train_groups_list = []

processed_train_files = set()

for _, row in train_df.iterrows():

    hdr_name = row["Subregions"]

    label = int(
        row["Label_6Class"]
    )

    sample_id = int(
        row["Sample_ID"]
    )

    hdr_path = find_hdr(
        hdr_name
    )

    if hdr_path is None:

        raise FileNotFoundError(
            f"Could not locate HSI file:\n{hdr_name}"
        )

    # Prevent duplicate processing
    normalized_path = os.path.abspath(
        hdr_path
    ).lower()

    if normalized_path in processed_train_files:
        continue

    processed_train_files.add(
        normalized_path
    )

    spectra = process_cube(
        hdr_path,
        label,
        sample_id
    )

    train_spectra_list.append(
        spectra
    )

    train_labels_list.extend(
        [label] * len(spectra)
    )

    train_groups_list.extend(
        [sample_id] * len(spectra)
    )


# ==============================================================
# COMBINE TRAINING DATA
# ==============================================================

X_train = np.vstack(
    train_spectra_list
)

y_train = np.asarray(
    train_labels_list
)

groups_train = np.asarray(
    train_groups_list
)

print()
print("=" * 70)
print("TRAINING MATRIX")
print("=" * 70)

print(
    "Training spectral matrix:",
    X_train.shape
)

print(
    "Training labels:",
    y_train.shape
)

print(
    "Training Sample_IDs:",
    len(np.unique(groups_train))
)

# ==============================================================
# PROCESS TESTING CUBES
# ==============================================================

print()
print("=" * 70)
print("PROCESSING TESTING CUBES")
print("=" * 70)

test_spectra_list = []
test_labels_list = []
test_groups_list = []

processed_test_files = set()

for _, row in test_df.iterrows():

    hdr_name = row["Subregions"]

    label = int(
        row["Label_6Class"]
    )

    sample_id = int(
        row["Sample_ID"]
    )

    hdr_path = find_hdr(
        hdr_name
    )

    if hdr_path is None:

        raise FileNotFoundError(
            f"Could not locate HSI file:\n{hdr_name}"
        )

    normalized_path = os.path.abspath(
        hdr_path
    ).lower()

    if normalized_path in processed_test_files:
        continue

    processed_test_files.add(
        normalized_path
    )

    spectra = process_cube(
        hdr_path,
        label,
        sample_id
    )

    test_spectra_list.append(
        spectra
    )

    test_labels_list.extend(
        [label] * len(spectra)
    )

    test_groups_list.extend(
        [sample_id] * len(spectra)
    )


# ==============================================================
# COMBINE TESTING DATA
# ==============================================================

X_test = np.vstack(
    test_spectra_list
)

y_test = np.asarray(
    test_labels_list
)

groups_test = np.asarray(
    test_groups_list
)

print()
print("=" * 70)
print("TESTING MATRIX")
print("=" * 70)

print(
    "Testing spectral matrix:",
    X_test.shape
)

print(
    "Testing labels:",
    y_test.shape
)

print(
    "Testing Sample_IDs:",
    len(np.unique(groups_test))
)

# ==============================================================
# LEAKAGE CHECK
# ==============================================================

print()
print("=" * 70)
print("HSI SOURCE SAMPLE LEAKAGE CHECK")
print("=" * 70)

train_groups = set(
    groups_train
)

test_groups = set(
    groups_test
)

overlap = train_groups.intersection(
    test_groups
)

print(
    "Training Sample_IDs:",
    len(train_groups)
)

print(
    "Testing Sample_IDs :",
    len(test_groups)
)

print(
    "Overlap            :",
    len(overlap)
)

if len(overlap) > 0:

    print()
    print("WARNING: SAMPLE_ID LEAKAGE DETECTED")

    raise ValueError(
        "Training and testing Sample_IDs overlap."
    )

else:

    print(
        "HSI leakage check: PASS"
    )

# ==============================================================
# NUMERICAL VALIDATION
# ==============================================================

print()
print("=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)

print(
    "Training NaN:",
    np.isnan(X_train).sum()
)

print(
    "Testing NaN :",
    np.isnan(X_test).sum()
)

print(
    "Training Inf:",
    np.isinf(X_train).sum()
)

print(
    "Testing Inf :",
    np.isinf(X_test).sum()
)

if not np.isfinite(X_train).all():
    raise ValueError(
        "Training data contains NaN/Inf."
    )

if not np.isfinite(X_test).all():
    raise ValueError(
        "Testing data contains NaN/Inf."
    )

# ==============================================================
# TRAINING-DERIVED MEAN CENTERING
# ==============================================================

print()
print("=" * 70)
print("TRAINING-DERIVED MEAN CENTERING")
print("=" * 70)

train_mean = np.mean(
    X_train,
    axis=0
)

X_train_centered = (
    X_train - train_mean
)

X_test_centered = (
    X_test - train_mean
)

print(
    "Training mean absolute mean:",
    np.mean(
        np.abs(
            np.mean(
                X_train_centered,
                axis=0
            )
        )
    )
)

print(
    "Test mean after training centering:",
    np.mean(
        X_test_centered
    )
)

# ==============================================================
# PCA
# ==============================================================

print()
print("=" * 70)
print("TRAINING PCA")
print("=" * 70)

print(
    "Original spectral bands:",
    X_train_centered.shape[1]
)

print(
    "Requested PCA components:",
    N_COMPONENTS
)

max_valid_components = min(
    X_train_centered.shape
)

print(
    "Maximum valid components:",
    max_valid_components
)

if N_COMPONENTS > max_valid_components:

    raise ValueError(
        f"Requested {N_COMPONENTS} PCs but maximum "
        f"valid number is {max_valid_components}."
    )

print()
print(
    "Fitting PCA using TRAINING data only..."
)

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=RANDOM_STATE
)

X_train_pca = pca.fit_transform(
    X_train_centered
)

print(
    "Transforming TEST data using training PCA..."
)

X_test_pca = pca.transform(
    X_test_centered
)

# ==============================================================
# PCA RESULTS
# ==============================================================

variance_retained = (
    np.sum(
        pca.explained_variance_ratio_
    ) * 100
)

print()
print(
    "Training PCA shape:",
    X_train_pca.shape
)

print(
    "Testing PCA shape :",
    X_test_pca.shape
)

print(
    f"Variance retained : "
    f"{variance_retained:.6f}%"
)

# ==============================================================
# CREATE PCA COLUMN NAMES
# ==============================================================

pca_columns = [
    f"PC{i}"
    for i in range(
        1,
        N_COMPONENTS + 1
    )
]

# ==============================================================
# CREATE TRAINING DATAFRAME
# ==============================================================

train_output = pd.DataFrame(
    X_train_pca,
    columns=pca_columns
)

train_output["label"] = y_train

train_output["Sample_ID"] = groups_train

# ==============================================================
# CREATE TESTING DATAFRAME
# ==============================================================

test_output = pd.DataFrame(
    X_test_pca,
    columns=pca_columns
)

test_output["label"] = y_test

test_output["Sample_ID"] = groups_test

# ==============================================================
# CREATE COMPLETE DATASET
# ==============================================================

complete_output = pd.concat(
    [
        train_output.assign(
            split="train"
        ),
        test_output.assign(
            split="test"
        )
    ],
    ignore_index=True
)

# ==============================================================
# SAVE DATASETS
# ==============================================================

train_output_path = os.path.join(
    OUTPUT_DIR,
    "papaya_train_90pc.csv"
)

test_output_path = os.path.join(
    OUTPUT_DIR,
    "papaya_test_90pc.csv"
)

complete_output_path = os.path.join(
    OUTPUT_DIR,
    "papaya_90pc_dataset.csv"
)

train_output.to_csv(
    train_output_path,
    index=False
)

test_output.to_csv(
    test_output_path,
    index=False
)

complete_output.to_csv(
    complete_output_path,
    index=False
)

# ==============================================================
# SAVE PCA INFORMATION
# ==============================================================

pca_variance_df = pd.DataFrame(
    {
        "PC": pca_columns,
        "Explained_Variance_Ratio":
            pca.explained_variance_ratio_,
        "Explained_Variance_Percent":
            pca.explained_variance_ratio_ * 100
    }
)

pca_variance_path = os.path.join(
    OUTPUT_DIR,
    "papaya_90pc_variance.csv"
)

pca_variance_df.to_csv(
    pca_variance_path,
    index=False
)

# ==============================================================
# SAVE SOURCE SPLIT INFORMATION
# ==============================================================

source_split_df = pd.DataFrame(
    {
        "Sample_ID":
            list(train_groups) +
            list(test_groups),

        "split":
            ["train"] * len(train_groups) +
            ["test"] * len(test_groups)
    }
)

source_split_path = os.path.join(
    OUTPUT_DIR,
    "papaya_90pc_source_split.csv"
)

source_split_df.to_csv(
    source_split_path,
    index=False
)

# ==============================================================
# CLASS DISTRIBUTION
# ==============================================================

print()
print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(
    pd.Series(y_train).value_counts().sort_index()
)

print()
print("=" * 70)
print("TESTING CLASS DISTRIBUTION")
print("=" * 70)

print(
    pd.Series(y_test).value_counts().sort_index()
)

# ==============================================================
# FINAL SUMMARY
# ==============================================================

print()
print("=" * 70)
print("PAPAYA 90-PC DATASET CREATED")
print("=" * 70)

print()
print("Original spectral bands :", X_train.shape[1])
print("PCA components          :", N_COMPONENTS)
print(
    f"Variance retained       : "
    f"{variance_retained:.6f}%"
)

print()
print("Training samples:", len(X_train_pca))
print("Testing samples :", len(X_test_pca))

print()
print(
    "Training Sample_IDs:",
    len(train_groups)
)

print(
    "Testing Sample_IDs :",
    len(test_groups)
)

print(
    "Sample_ID overlap  :",
    len(overlap)
)

print()
print("Saved files:")

print(
    train_output_path
)

print(
    test_output_path
)

print(
    complete_output_path
)

print(
    pca_variance_path
)

print(
    source_split_path
)

print()
print("=" * 70)
print("PAPAYA FINAL 90-PC PREPARATION COMPLETED")
print("=" * 70)

print()
print(
    "Next step:"
)

print(
    "Run final Random Forest using the 90 PCA components."
)

print(
    "After that, run SVM using the same 90-PC data."
)

print("=" * 70)