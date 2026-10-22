# ================================================================
# EGGPLANT + CABBAGE HSI PREPARATION
# IMPROVED / MEMORY-EFFICIENT VERSION
# ================================================================
#
# Purpose:
#   Prepare Eggplant and Cabbage HSI datasets for:
#
#       Spatial Split
#           ↓
#       Spectral Preprocessing
#           ↓
#       PCA
#           ↓
#       Random Forest
#           ↓
#       SVM
#
# IMPORTANT:
#   We DO NOT save millions of pixels as CSV.
#   Instead, a balanced spatially separated sample is created.
#
# Classes:
#   1 = Low
#   2 = Medium
#   3 = High
#
# 0 = Unclassified -> removed
#
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spectral import open_image


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral"
)

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
# DATASET LOCATIONS
# ================================================================

DATASETS = {

    "Eggplant": {
        "directory": os.path.join(
            RAW_DIR,
            "eggplant_hsi",
            "Eggplant_Crop"
        ),

        "reflectance": os.path.join(
            RAW_DIR,
            "eggplant_hsi",
            "Eggplant_Crop",
            "Eggplant_Reflectance_Data.hdr"
        ),

        "gt": os.path.join(
            RAW_DIR,
            "eggplant_hsi",
            "Eggplant_Crop",
            "Eggplant_N2_Concentration_GT.hdr"
        )
    },

    "Cabbage": {
        "directory": os.path.join(
            RAW_DIR,
            "cabbage_hsi",
            "Cabbage_Crop"
        ),

        "reflectance": os.path.join(
            RAW_DIR,
            "cabbage_hsi",
            "Cabbage_Crop",
            "Cabbage_Reflectance_Data.hdr"
        ),

        "gt": os.path.join(
            RAW_DIR,
            "cabbage_hsi",
            "Cabbage_Crop",
            "Cabbage_N2_Concentration_GT.hdr"
        )
    }
}


# ================================================================
# OUTPUT DIRECTORIES
# ================================================================

OUTPUTS = {}

for dataset_name in DATASETS:

    output_dir = os.path.join(
        PROCESSED_DIR,
        f"{dataset_name.lower()}_classification"
    )

    graph_dir = os.path.join(
        GRAPH_DIR,
        f"{dataset_name.lower()}_classification"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    OUTPUTS[dataset_name] = {
        "output": output_dir,
        "graph": graph_dir
    }


# ================================================================
# SETTINGS
# ================================================================

RANDOM_STATE = 42

# Number of pixels per class used for the final datasets.
#
# This is deliberately much smaller than the millions of available
# pixels.
#
# 10,000/class/train
# 10,000/class/test
#
# Total:
#   30,000 training pixels
#   30,000 testing pixels
#
SAMPLES_PER_CLASS_TRAIN = 10000
SAMPLES_PER_CLASS_TEST = 10000

# Spatial block size.
#
# Pixels inside one block belong to the same spatial group.
# Blocks are assigned completely to either train or test.
BLOCK_ROWS = 20
BLOCK_COLS = 100

TRAIN_BLOCK_FRACTION = 0.70


# ================================================================
# CLASS NAMES
# ================================================================

CLASS_NAMES = {
    0: "Unclassified",
    1: "Low",
    2: "Medium",
    3: "High"
}


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("EGGPLANT + CABBAGE HSI PREPARATION")
print("IMPROVED MEMORY-EFFICIENT VERSION")
print("=" * 70)

print("""
IMPORTANT CHANGE:

The previous script attempted to write millions of full
spectral pixels to CSV.

That is unnecessary and extremely slow.

This version:

1. Reads the HSI reflectance cube
2. Reads the GT image
3. Removes unclassified pixels
4. Creates spatial blocks
5. Separates blocks into train/test
6. Samples a balanced number of pixels
7. Saves compact NPZ data
8. Saves manageable CSV files
9. Creates spectral signature graphs

The original raw HSI data is never modified.
""")


# ================================================================
# WAVELENGTH FUNCTION
# ================================================================

def get_wavelengths(image):

    metadata = image.metadata

    for key in [
        "wavelength",
        "wavelengths"
    ]:

        if key in metadata:

            try:

                values = metadata[key]

                values = np.array(
                    [
                        float(x)
                        for x in values
                    ],
                    dtype=np.float32
                )

                return values

            except Exception:

                pass

    return None


# ================================================================
# NORMALIZE REFLECTANCE
# ================================================================

def normalize_reflectance(X):

    X = np.asarray(
        X,
        dtype=np.float32
    )

    finite_values = X[
        np.isfinite(X)
    ]

    if len(finite_values) == 0:

        raise ValueError(
            "No finite reflectance values found."
        )

    minimum = float(
        np.percentile(
            finite_values,
            1
        )
    )

    maximum = float(
        np.percentile(
            finite_values,
            99
        )
    )

    print(
        f"Reflectance percentile range: "
        f"{minimum:.6f} - {maximum:.6f}"
    )

    # ------------------------------------------------------------
    # Detect common scaling
    # ------------------------------------------------------------

    absolute_max = float(
        np.max(
            finite_values
        )
    )

    if absolute_max > 1000:

        print(
            "Detected large-scale reflectance values."
        )

        print(
            "Dividing by 4095."
        )

        X = X / 4095.0

    elif absolute_max > 10:

        print(
            "Detected scaled reflectance values."
        )

        X = X / absolute_max

    else:

        print(
            "Reflectance appears to be 0-1-like."
        )

    return X


# ================================================================
# CREATE SPATIAL BLOCKS
# ================================================================

def create_spatial_blocks(
    height,
    width
):

    block_rows = int(
        np.ceil(
            height / BLOCK_ROWS
        )
    )

    block_cols = int(
        np.ceil(
            width / BLOCK_COLS
        )
    )

    block_ids = np.empty(
        (
            height,
            width
        ),
        dtype=np.int32
    )

    block_id = 0

    for br in range(
        block_rows
    ):

        r_start = (
            br * BLOCK_ROWS
        )

        r_end = min(
            r_start + BLOCK_ROWS,
            height
        )

        for bc in range(
            block_cols
        ):

            c_start = (
                bc * BLOCK_COLS
            )

            c_end = min(
                c_start + BLOCK_COLS,
                width
            )

            block_ids[
                r_start:r_end,
                c_start:c_end
            ] = block_id

            block_id += 1

    return block_ids, block_id


# ================================================================
# SELECT TRAIN / TEST BLOCKS
# ================================================================

def select_spatial_blocks(
    gt,
    block_ids,
    number_of_blocks
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    block_classes = {}

    for block_id in range(
        number_of_blocks
    ):

        mask = (
            block_ids == block_id
        )

        values = gt[
            mask
        ]

        valid = values[
            np.isin(
                values,
                [1, 2, 3]
            )
        ]

        if len(valid) == 0:

            continue

        block_classes[
            block_id
        ] = set(
            np.unique(
                valid
            ).astype(int)
        )

    block_list = list(
        block_classes.keys()
    )

    rng.shuffle(
        block_list
    )

    target_blocks = int(
        len(block_list)
        * TRAIN_BLOCK_FRACTION
    )

    train_blocks = []
    test_blocks = []

    # ------------------------------------------------------------
    # Greedy class-aware spatial assignment
    # ------------------------------------------------------------

    train_class_presence = {
        1: 0,
        2: 0,
        3: 0
    }

    test_class_presence = {
        1: 0,
        2: 0,
        3: 0
    }

    for block in block_list:

        classes = block_classes[
            block
        ]

        train_score = sum(
            1
            for c in classes
            if train_class_presence[c] == 0
        )

        test_score = sum(
            1
            for c in classes
            if test_class_presence[c] == 0
        )

        if len(train_blocks) < target_blocks:

            if (
                train_score >= test_score
                or len(test_blocks) == 0
            ):

                train_blocks.append(
                    block
                )

                for c in classes:
                    train_class_presence[c] += 1

                continue

        test_blocks.append(
            block
        )

        for c in classes:
            test_class_presence[c] += 1

    return (
        set(train_blocks),
        set(test_blocks)
    )


# ================================================================
# SAMPLE PIXELS FROM SPATIAL GROUP
# ================================================================

def sample_pixels(
    X,
    y,
    rows,
    cols,
    allowed_blocks,
    block_ids,
    samples_per_class,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    selected_indices = []

    for class_value in [1, 2, 3]:

        class_mask = (
            y == class_value
        )

        block_mask = np.isin(
            block_ids,
            list(allowed_blocks)
        )

        combined_mask = (
            class_mask
            &
            block_mask
        )

        indices = np.flatnonzero(
            combined_mask
        )

        print(
            f"Class {class_value} "
            f"({CLASS_NAMES[class_value]}): "
            f"{len(indices):,} available"
        )

        if len(indices) == 0:

            raise ValueError(
                f"No samples available "
                f"for class {class_value} "
                f"in this spatial split."
            )

        number_to_select = min(
            samples_per_class,
            len(indices)
        )

        selected = rng.choice(
            indices,
            size=number_to_select,
            replace=False
        )

        selected_indices.extend(
            selected.tolist()
        )

    selected_indices = np.array(
        selected_indices,
        dtype=np.int64
    )

    rng.shuffle(
        selected_indices
    )

    return selected_indices


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

    reflectance_path = (
        config["reflectance"]
    )

    gt_path = config["gt"]

    output_dir = (
        OUTPUTS[dataset_name]["output"]
    )

    graph_dir = (
        OUTPUTS[dataset_name]["graph"]
    )

    # ============================================================
    # FILE VALIDATION
    # ============================================================

    print("\nFILE CHECK")
    print("-" * 70)

    if not os.path.exists(
        reflectance_path
    ):

        raise FileNotFoundError(
            f"Reflectance file not found:\n"
            f"{reflectance_path}"
        )

    if not os.path.exists(
        gt_path
    ):

        raise FileNotFoundError(
            f"GT file not found:\n"
            f"{gt_path}"
        )

    print(
        "Reflectance file: FOUND"
    )

    print(
        "GT file         : FOUND"
    )

    # ============================================================
    # LOAD HSI
    # ============================================================

    print("\nLOADING REFLECTANCE HSI")
    print("-" * 70)

    hsi_image = open_image(
        reflectance_path
    )

    print(
        "HSI shape:",
        hsi_image.shape
    )

    wavelengths = get_wavelengths(
        hsi_image
    )

    if wavelengths is not None:

        print(
            "Wavelength count:",
            len(wavelengths)
        )

        print(
            "First wavelength:",
            wavelengths[0]
        )

        print(
            "Last wavelength:",
            wavelengths[-1]
        )

    else:

        print(
            "WARNING: Wavelength metadata "
            "not found."
        )

    # ============================================================
    # LOAD GT
    # ============================================================

    print("\nLOADING GROUND TRUTH")
    print("-" * 70)

    gt_image = open_image(
        gt_path
    )

    print(
        "GT shape:",
        gt_image.shape
    )

    # ============================================================
    # LOAD ARRAYS
    # ============================================================

    print("\nLOADING ARRAYS")
    print("-" * 70)

    hsi = np.asarray(
        hsi_image.load(),
        dtype=np.float32
    )

    gt = np.asarray(
        gt_image.load()
    )

    if gt.ndim == 3:

        if gt.shape[2] == 1:

            gt = gt[:, :, 0]

        else:

            raise ValueError(
                "GT contains more than one band."
            )

    print(
        "Loaded HSI:",
        hsi.shape
    )

    print(
        "Loaded GT:",
        gt.shape
    )

    # ============================================================
    # ALIGNMENT CHECK
    # ============================================================

    print("\nSPATIAL ALIGNMENT")
    print("-" * 70)

    if (
        hsi.shape[0] != gt.shape[0]
        or
        hsi.shape[1] != gt.shape[1]
    ):

        raise ValueError(
            "HSI and GT spatial dimensions "
            "do not match.\n"
            f"HSI: {hsi.shape}\n"
            f"GT : {gt.shape}"
        )

    print(
        "Spatial alignment: PASS"
    )

    height = hsi.shape[0]
    width = hsi.shape[1]
    bands = hsi.shape[2]

    print(
        "Height:",
        height
    )

    print(
        "Width:",
        width
    )

    print(
        "Bands:",
        bands
    )

    # ============================================================
    # GT DISTRIBUTION
    # ============================================================

    print("\nGROUND TRUTH DISTRIBUTION")
    print("-" * 70)

    unique_values, counts = np.unique(
        gt,
        return_counts=True
    )

    for value, count in zip(
        unique_values,
        counts
    ):

        name = CLASS_NAMES.get(
            int(value),
            "Unknown"
        )

        print(
            f"{int(value)} "
            f"({name}): "
            f"{int(count):,}"
        )

    # ============================================================
    # NUMERICAL VALIDATION
    # ============================================================

    print("\nNUMERICAL VALIDATION")
    print("-" * 70)

    nan_count = int(
        np.isnan(hsi).sum()
    )

    inf_count = int(
        np.isinf(hsi).sum()
    )

    print(
        "NaN:",
        nan_count
    )

    print(
        "Inf:",
        inf_count
    )

    # ============================================================
    # REFLECTANCE SCALING
    # ============================================================

    print("\nREFLECTANCE SCALING")
    print("-" * 70)

    hsi = normalize_reflectance(
        hsi
    )

    # ============================================================
    # CREATE LABEL MAP
    # ============================================================

    print("\nCREATING LABEL MAP")
    print("-" * 70)

    valid_mask = np.isin(
        gt,
        [1, 2, 3]
    )

    print(
        "Valid pixels:",
        f"{int(valid_mask.sum()):,}"
    )

    print(
        "Unclassified:",
        f"{int((~valid_mask).sum()):,}"
    )

    # ============================================================
    # CREATE SPATIAL BLOCKS
    # ============================================================

    print("\nCREATING SPATIAL BLOCKS")
    print("-" * 70)

    block_ids, number_of_blocks = (
        create_spatial_blocks(
            height,
            width
        )
    )

    print(
        "Block rows:",
        BLOCK_ROWS
    )

    print(
        "Block columns:",
        BLOCK_COLS
    )

    print(
        "Total spatial blocks:",
        number_of_blocks
    )

    # ============================================================
    # SELECT TRAIN / TEST BLOCKS
    # ============================================================

    print("\nCREATING SPATIAL TRAIN / TEST SPLIT")
    print("-" * 70)

    train_blocks, test_blocks = (
        select_spatial_blocks(
            gt,
            block_ids,
            number_of_blocks
        )
    )

    overlap = (
        train_blocks
        &
        test_blocks
    )

    print(
        "Training blocks:",
        len(train_blocks)
    )

    print(
        "Testing blocks:",
        len(test_blocks)
    )

    print(
        "Overlapping blocks:",
        len(overlap)
    )

    if len(overlap) != 0:

        raise ValueError(
            "Spatial block leakage detected."
        )

    print(
        "Spatial block leakage: PASS"
    )

    # ============================================================
    # CREATE PIXEL INDEX MAP
    # ============================================================

    rows_grid, cols_grid = np.indices(
        (height, width)
    )

    flat_gt = gt.reshape(
        -1
    )

    flat_blocks = block_ids.reshape(
        -1
    )

    flat_rows = rows_grid.reshape(
        -1
    )

    flat_cols = cols_grid.reshape(
        -1
    )

    # ============================================================
    # SAMPLE TRAINING PIXELS
    # ============================================================

    print("\nSAMPLING TRAINING PIXELS")
    print("-" * 70)

    train_indices = sample_pixels(
        flat_gt,
        flat_gt,
        flat_rows,
        flat_cols,
        train_blocks,
        flat_blocks,
        SAMPLES_PER_CLASS_TRAIN,
        RANDOM_STATE
    )

    # ============================================================
    # SAMPLE TEST PIXELS
    # ============================================================

    print("\nSAMPLING TESTING PIXELS")
    print("-" * 70)

    test_indices = sample_pixels(
        flat_gt,
        flat_gt,
        flat_rows,
        flat_cols,
        test_blocks,
        flat_blocks,
        SAMPLES_PER_CLASS_TEST,
        RANDOM_STATE + 100
    )

    # ============================================================
    # EXTRACT SPECTRA
    # ============================================================

    print("\nEXTRACTING TRAINING SPECTRA")
    print("-" * 70)

    hsi_flat = hsi.reshape(
        -1,
        bands
    )

    X_train = hsi_flat[
        train_indices
    ]

    X_test = hsi_flat[
        test_indices
    ]

    y_train = flat_gt[
        train_indices
    ].astype(np.int64)

    y_test = flat_gt[
        test_indices
    ].astype(np.int64)

    rows_train = flat_rows[
        train_indices
    ]

    cols_train = flat_cols[
        train_indices
    ]

    rows_test = flat_rows[
        test_indices
    ]

    cols_test = flat_cols[
        test_indices
    ]

    print(
        "Training matrix:",
        X_train.shape
    )

    print(
        "Testing matrix:",
        X_test.shape
    )

    # ============================================================
    # FINAL NUMERICAL CHECK
    # ============================================================

    print("\nFINAL NUMERICAL VALIDATION")
    print("-" * 70)

    print(
        "Train NaN:",
        int(np.isnan(X_train).sum())
    )

    print(
        "Test NaN:",
        int(np.isnan(X_test).sum())
    )

    print(
        "Train Inf:",
        int(np.isinf(X_train).sum())
    )

    print(
        "Test Inf:",
        int(np.isinf(X_test).sum())
    )

    # ============================================================
    # CLASS DISTRIBUTION
    # ============================================================

    print("\nTRAINING CLASS DISTRIBUTION")
    print("-" * 70)

    train_distribution = (
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    print(
        train_distribution
    )

    print("\nTESTING CLASS DISTRIBUTION")
    print("-" * 70)

    test_distribution = (
        pd.Series(y_test)
        .value_counts()
        .sort_index()
    )

    print(
        test_distribution
    )

    # ============================================================
    # SAVE NPZ
    # ============================================================

    print("\nSAVING COMPACT NPZ DATA")
    print("-" * 70)

    npz_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_spatial_classification.npz"
    )

    np.savez_compressed(
        npz_file,
        X_train=X_train.astype(
            np.float32
        ),
        X_test=X_test.astype(
            np.float32
        ),
        y_train=y_train,
        y_test=y_test,
        rows_train=rows_train,
        cols_train=cols_train,
        rows_test=rows_test,
        cols_test=cols_test
    )

    print(
        "Saved:",
        npz_file
    )

    # ============================================================
    # SAVE MANAGEABLE CSV
    # ============================================================

    print("\nSAVING MANAGEABLE CSV FILES")
    print("-" * 70)

    band_columns = [
        f"Band_{i+1}"
        for i in range(bands)
    ]

    train_df = pd.DataFrame(
        X_train,
        columns=band_columns
    )

    train_df.insert(
        0,
        "label",
        y_train
    )

    train_df.insert(
        1,
        "class_name",
        [
            CLASS_NAMES[int(x)]
            for x in y_train
        ]
    )

    train_df.insert(
        2,
        "row",
        rows_train
    )

    train_df.insert(
        3,
        "column",
        cols_train
    )

    test_df = pd.DataFrame(
        X_test,
        columns=band_columns
    )

    test_df.insert(
        0,
        "label",
        y_test
    )

    test_df.insert(
        1,
        "class_name",
        [
            CLASS_NAMES[int(x)]
            for x in y_test
        ]
    )

    test_df.insert(
        2,
        "row",
        rows_test
    )

    test_df.insert(
        3,
        "column",
        cols_test
    )

    train_csv = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_train_spectral.csv"
    )

    test_csv = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_test_spectral.csv"
    )

    train_df.to_csv(
        train_csv,
        index=False
    )

    test_df.to_csv(
        test_csv,
        index=False
    )

    print(
        "Training CSV:",
        train_csv
    )

    print(
        "Testing CSV:",
        test_csv
    )

    # ============================================================
    # SAVE WAVELENGTHS
    # ============================================================

    if wavelengths is not None:

        wavelength_df = pd.DataFrame({

            "band":
                np.arange(
                    1,
                    len(wavelengths) + 1
                ),

            "wavelength_nm":
                wavelengths
        })

        wavelength_file = os.path.join(
            output_dir,
            f"{dataset_name.lower()}_wavelengths.csv"
        )

        wavelength_df.to_csv(
            wavelength_file,
            index=False
        )

        print(
            "Wavelengths:",
            wavelength_file
        )

    # ============================================================
    # CLASS SPECTRAL SIGNATURES
    # ============================================================

    print("\nCALCULATING CLASS SPECTRAL SIGNATURES")
    print("-" * 70)

    signature_rows = []

    for class_value in [1, 2, 3]:

        mask = (
            y_train == class_value
        )

        class_data = X_train[
            mask
        ]

        mean_spectrum = np.mean(
            class_data,
            axis=0
        )

        std_spectrum = np.std(
            class_data,
            axis=0
        )

        for band_index in range(
            bands
        ):

            wavelength = (
                wavelengths[band_index]
                if wavelengths is not None
                else np.nan
            )

            signature_rows.append({

                "label":
                    class_value,

                "class_name":
                    CLASS_NAMES[
                        class_value
                    ],

                "band":
                    band_index + 1,

                "wavelength_nm":
                    wavelength,

                "mean":
                    mean_spectrum[
                        band_index
                    ],

                "std":
                    std_spectrum[
                        band_index
                    ]
            })

    signature_df = pd.DataFrame(
        signature_rows
    )

    signature_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_spectral_signatures.csv"
    )

    signature_df.to_csv(
        signature_file,
        index=False
    )

    print(
        "Spectral signatures:",
        signature_file
    )

    # ============================================================
    # SPECTRAL SIGNATURE GRAPH
    # ============================================================

    plt.figure(
        figsize=(12, 7)
    )

    for class_value in [1, 2, 3]:

        data = signature_df[
            signature_df["label"]
            == class_value
        ]

        if wavelengths is not None:

            x = data[
                "wavelength_nm"
            ]

        else:

            x = data[
                "band"
            ]

        plt.plot(
            x,
            data["mean"],
            label=CLASS_NAMES[
                class_value
            ]
        )

    plt.title(
        f"{dataset_name} HSI "
        "Class Spectral Signatures"
    )

    plt.xlabel(
        "Wavelength (nm)"
        if wavelengths is not None
        else "Spectral Band"
    )

    plt.ylabel(
        "Mean Reflectance"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    signature_graph = os.path.join(
        graph_dir,
        f"{dataset_name.lower()}_spectral_signatures.png"
    )

    plt.savefig(
        signature_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Signature graph:",
        signature_graph
    )

    # ============================================================
    # GT MAP
    # ============================================================

    plt.figure(
        figsize=(12, 7)
    )

    plt.imshow(
        gt
    )

    plt.title(
        f"{dataset_name} "
        "Ground Truth Classification"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Row"
    )

    plt.colorbar()

    plt.tight_layout()

    gt_graph = os.path.join(
        graph_dir,
        f"{dataset_name.lower()}_gt_map.png"
    )

    plt.savefig(
        gt_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "GT graph:",
        gt_graph
    )

    # ============================================================
    # SPLIT VISUALIZATION
    # ============================================================

    split_map = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    train_mask = np.isin(
        block_ids,
        list(train_blocks)
    )

    test_mask = np.isin(
        block_ids,
        list(test_blocks)
    )

    split_map[
        train_mask
    ] = 1

    split_map[
        test_mask
    ] = 2

    split_graph = os.path.join(
        graph_dir,
        f"{dataset_name.lower()}_spatial_train_test_split.png"
    )

    plt.figure(
        figsize=(12, 7)
    )

    plt.imshow(
        split_map
    )

    plt.title(
        f"{dataset_name} "
        "Spatial Train/Test Split"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Row"
    )

    plt.colorbar(
        ticks=[0, 1, 2],
        label="0=Unused, 1=Train, 2=Test"
    )

    plt.tight_layout()

    plt.savefig(
        split_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Split graph:",
        split_graph
    )

    # ============================================================
    # SAVE SUMMARY
    # ============================================================

    summary = {

        "dataset":
            dataset_name,

        "height":
            height,

        "width":
            width,

        "bands":
            bands,

        "total_pixels":
            int(gt.size),

        "valid_pixels":
            int(valid_mask.sum()),

        "training_samples":
            int(len(X_train)),

        "testing_samples":
            int(len(X_test)),

        "training_blocks":
            len(train_blocks),

        "testing_blocks":
            len(test_blocks),

        "overlapping_blocks":
            len(overlap),

        "train_low":
            int(np.sum(y_train == 1)),

        "train_medium":
            int(np.sum(y_train == 2)),

        "train_high":
            int(np.sum(y_train == 3)),

        "test_low":
            int(np.sum(y_test == 1)),

        "test_medium":
            int(np.sum(y_test == 2)),

        "test_high":
            int(np.sum(y_test == 3)),

        "nan_values":
            nan_count,

        "inf_values":
            inf_count
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_file = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_dataset_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print(
        "\nSummary:",
        summary_file
    )

    # ============================================================
    # DELETE REFERENCES TO LARGE ARRAYS
    # ============================================================

    del hsi
    del hsi_flat
    del gt
    del X_train
    del X_test

    # ============================================================
    # FINAL
    # ============================================================

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} PREPARATION COMPLETED"
    )
    print("=" * 70)

    print(
        f"""
Training samples : {len(train_indices):,}
Testing samples  : {len(test_indices):,}

Samples per class:
Training = up to {SAMPLES_PER_CLASS_TRAIN:,}
Testing  = up to {SAMPLES_PER_CLASS_TEST:,}

Spatial block leakage:
PASS

NPZ:
{npz_file}

Training CSV:
{train_csv}

Testing CSV:
{test_csv}
"""
    )

    return summary


# ================================================================
# PROCESS BOTH DATASETS
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
print("EGGPLANT + CABBAGE FINAL SUMMARY")
print("=" * 70)

combined_df = pd.DataFrame(
    results
)

print(
    combined_df.to_string(
        index=False
    )
)

combined_file = os.path.join(
    PROCESSED_DIR,
    "eggplant_cabbage_preparation_summary.csv"
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
# FINAL MESSAGE
# ================================================================

print("\n" + "=" * 70)
print("EGGPLANT + CABBAGE HSI PREPARATION COMPLETED")
print("=" * 70)

print("""
IMPORTANT:

We intentionally did NOT create a CSV containing every valid
pixel from the original HSI cubes.

Instead we created:

    Spatially separated training samples
                    +
    Spatially separated testing samples
                    +
    Balanced classes
                    +
    Original pixel coordinates
                    +
    Compact NPZ files
                    +
    Manageable CSV files

NEXT:

    Eggplant ──┐
               ├── Spectral preprocessing
    Cabbage ───┘
                    ↓
                  PCA
                    ↓
          PCA component sensitivity
                    ↓
          Select optimal components
                    ↓
            Random Forest
                    ↓
                  SVM
                    ↓
             Model comparison
""")

print("=" * 70)