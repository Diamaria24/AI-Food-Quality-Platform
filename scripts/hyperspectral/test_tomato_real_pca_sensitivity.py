"""
test_tomato_real_pca_sensitivity.py

Tomato HSI PCA sensitivity experiment.

Original data:
    550 spectral bands
    400-950 nm

Classes:
    Healthy
    Alternaria alternata
    Alternaria solani
    Botrytis cinerea
    Fusarium oxysporum

IMPORTANT:
    PCA is fitted ONLY on training cubes.

    Test cubes remain completely unseen during:
        - preprocessing statistics
        - PCA fitting
        - model training

Experiments:
    2, 10, 20, 30, 40, 50 PCA components

The 20-PC train/test datasets are additionally saved for
the final Random Forest analysis.
"""

import os
import warnings

import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


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

TOMATO_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "tomato_hsi"
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

PC20_DIR = os.path.join(
    CLASSIFICATION_DIR,
    "pca_20pc"
)

GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "tomato_classification"
)


os.makedirs(
    SENSITIVITY_DIR,
    exist_ok=True
)

os.makedirs(
    PC20_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

WAVELENGTH_START = 400
WAVELENGTH_END = 950

EXPECTED_BANDS = 550

MAX_PCA_COMPONENTS = 50

COMPONENTS_TO_TEST = [
    2,
    10,
    20,
    30,
    40,
    50
]

BEST_COMPONENTS = 20

SAMPLES_PER_CUBE = 5000

SAVGOL_WINDOW = 7

SAVGOL_POLYORDER = 2

RF_TREES = 200

RANDOM_STATE = 42


CLASS_ORDER = [
    "Healthy",
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum"
]


# ============================================================
# CLASS / FILE MAPPING
# ============================================================

CLASS_FILES = {

    "Healthy": [
        "Healthy_1.mat",
        "Healthy_2.mat"
    ],

    "Alternaria alternata": [
        "Alternaria_alternata_1.mat",
        "Alternaria_alternata_2.mat"
    ],

    "Alternaria solani": [
        "Alternaria_solani_1.mat",
        "Alternaria_solani_2.mat"
    ],

    "Botrytis cinerea": [
        "Botrytis_cinerea_1.mat",
        "Botrytis_cinerea_2.mat"
    ],

    "Fusarium oxysporum": [
        "Fusarium_oxysporum_1.mat",
        "Fusarium_oxysporum_2.mat"
    ]
}


# ============================================================
# PRINT HEADER
# ============================================================

def header(title):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# FIND MAT VARIABLE
# ============================================================

def find_cube_variable(mat_data):

    candidates = []

    for key, value in mat_data.items():

        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray):

            if value.ndim == 3:

                candidates.append(
                    (key, value)
                )

    if not candidates:

        raise ValueError(
            "No 3-dimensional hyperspectral cube "
            "was found in the MAT file."
        )

    # Prefer Day_one_* variables
    for key, value in candidates:

        if key.startswith("Day_one"):

            return key, value

    return candidates[0]


# ============================================================
# LOAD MAT CUBE
# ============================================================

def load_cube(filename):

    path = os.path.join(
        TOMATO_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"MAT file not found:\n{path}"
        )

    print()
    print(
        f"Loading: {filename}"
    )

    mat_data = loadmat(
        path
    )

    variable_name, cube = (
        find_cube_variable(
            mat_data
        )
    )

    cube = np.asarray(
        cube,
        dtype=np.float32
    )

    print(
        f"Variable : {variable_name}"
    )

    print(
        f"Shape    : {cube.shape}"
    )

    print(
        f"Dtype    : {cube.dtype}"
    )

    if cube.ndim != 3:

        raise ValueError(
            f"Expected 3D cube, got "
            f"{cube.ndim} dimensions."
        )

    if cube.shape[2] != EXPECTED_BANDS:

        raise ValueError(
            f"Expected {EXPECTED_BANDS} spectral bands, "
            f"got {cube.shape[2]}."
        )

    return cube


# ============================================================
# SCALE NORMALIZATION
# ============================================================

def normalize_scale(cube):

    cube = np.asarray(
        cube,
        dtype=np.float32
    )

    finite_values = cube[
        np.isfinite(cube)
    ]

    if finite_values.size == 0:

        raise ValueError(
            "Cube contains no finite values."
        )

    maximum = float(
        np.max(finite_values)
    )

    minimum = float(
        np.min(finite_values)
    )

    if maximum > 100:

        print(
            "Scale    : 4095-like -> divided by 4095"
        )

        cube = cube / 4095.0

    else:

        print(
            "Scale    : 0-1-like -> unchanged"
        )

    return cube.astype(
        np.float32
    )


# ============================================================
# CLEAN INVALID VALUES
# ============================================================

def clean_values(cube):

    cube = np.nan_to_num(
        cube,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return cube.astype(
        np.float32
    )


# ============================================================
# SPECTRAL SMOOTHING
# ============================================================

def smooth_cube(cube):

    print(
        f"Smoothing: window={SAVGOL_WINDOW}, "
        f"polyorder={SAVGOL_POLYORDER}"
    )

    smoothed = savgol_filter(
        cube,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER,
        axis=2
    )

    return smoothed.astype(
        np.float32
    )


# ============================================================
# PREPROCESS CUBE
# ============================================================

def preprocess_cube(cube):

    cube = normalize_scale(
        cube
    )

    cube = clean_values(
        cube
    )

    cube = smooth_cube(
        cube
    )

    return cube


# ============================================================
# SAMPLE PIXELS
# ============================================================

def sample_cube_pixels(
    cube,
    n_samples,
    random_state
):

    height, width, bands = (
        cube.shape
    )

    pixels = cube.reshape(
        -1,
        bands
    )

    # --------------------------------------------------------
    # Valid pixel check
    # --------------------------------------------------------

    valid_mask = np.isfinite(
        pixels
    ).all(
        axis=1
    )

    pixels = pixels[
        valid_mask
    ]

    print(
        f"Valid pixels: {len(pixels)}"
    )

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    if len(pixels) <= n_samples:

        sampled = pixels

        print(
            f"Sampled pixels: "
            f"{len(sampled)}"
        )

        return sampled

    rng = np.random.default_rng(
        random_state
    )

    indices = rng.choice(
        len(pixels),
        size=n_samples,
        replace=False
    )

    sampled = pixels[
        indices
    ]

    print(
        f"Sampled pixels: "
        f"{len(sampled)}"
    )

    return sampled


# ============================================================
# BUILD TRAIN / TEST FILE LIST
# ============================================================

def create_split():

    header(
        "CUBE-LEVEL TRAIN / TEST SPLIT"
    )

    train_files = []
    test_files = []

    for class_name in CLASS_ORDER:

        files = CLASS_FILES[
            class_name
        ]

        train_file = files[0]
        test_file = files[1]

        train_files.append(
            (
                train_file,
                class_name
            )
        )

        test_files.append(
            (
                test_file,
                class_name
            )
        )

        print()
        print(
            class_name
        )

        print(
            f"  TRAIN: {train_file}"
        )

        print(
            f"  TEST : {test_file}"
        )

    return (
        train_files,
        test_files
    )


# ============================================================
# PROCESS FILE LIST
# ============================================================

def process_files(
    file_list,
    description
):

    header(
        f"PROCESSING {description.upper()} CUBES"
    )

    all_samples = []
    all_labels = []
    all_sources = []

    for file_index, (
        filename,
        class_name
    ) in enumerate(
        file_list
    ):

        cube = load_cube(
            filename
        )

        cube = preprocess_cube(
            cube
        )

        samples = sample_cube_pixels(
            cube,
            SAMPLES_PER_CUBE,
            RANDOM_STATE + file_index
        )

        all_samples.append(
            samples
        )

        all_labels.extend(
            [class_name] * len(samples)
        )

        all_sources.extend(
            [filename] * len(samples)
        )

    X = np.vstack(
        all_samples
    ).astype(
        np.float32
    )

    y = np.asarray(
        all_labels
    )

    sources = np.asarray(
        all_sources
    )

    return (
        X,
        y,
        sources
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

def leakage_check(
    train_sources,
    test_sources
):

    header(
        "SOURCE-CUBE LEAKAGE CHECK"
    )

    train_set = set(
        train_sources
    )

    test_set = set(
        test_sources
    )

    overlap = (
        train_set.intersection(
            test_set
        )
    )

    print(
        f"Training cubes: "
        f"{len(train_set)}"
    )

    print(
        f"Testing cubes : "
        f"{len(test_set)}"
    )

    print(
        f"Overlap       : "
        f"{len(overlap)}"
    )

    if overlap:

        for item in sorted(
            overlap
        ):

            print(
                f" - {item}"
            )

        raise ValueError(
            "SOURCE-CUBE LEAKAGE DETECTED."
        )

    print(
        "Leakage check: PASS"
    )


# ============================================================
# TRAINING-DERIVED MEAN CENTERING
# ============================================================

def mean_center(
    X_train,
    X_test
):

    header(
        "TRAINING-DERIVED MEAN CENTERING"
    )

    training_mean = np.mean(
        X_train,
        axis=0
    )

    X_train_centered = (
        X_train
        -
        training_mean
    )

    X_test_centered = (
        X_test
        -
        training_mean
    )

    train_mean_error = np.mean(
        np.abs(
            np.mean(
                X_train_centered,
                axis=0
            )
        )
    )

    test_mean_after = np.mean(
        X_test_centered
    )

    print(
        "Training mean absolute mean: "
        f"{train_mean_error:.10f}"
    )

    print(
        "Test mean after training centering: "
        f"{test_mean_after:.10f}"
    )

    return (
        X_train_centered.astype(np.float32),
        X_test_centered.astype(np.float32),
        training_mean
    )


# ============================================================
# FIT PCA
# ============================================================

def fit_pca(
    X_train,
    X_test,
    n_components
):

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

    variance = (
        np.sum(
            pca.explained_variance_ratio_
        )
        *
        100
    )

    return (
        X_train_pca.astype(np.float32),
        X_test_pca.astype(np.float32),
        pca,
        variance
    )


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train
):

    model = RandomForestClassifier(
        n_estimators=RF_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# SAVE 20-PC DATASET
# ============================================================

def save_20pc_dataset(
    X_train_pca,
    X_test_pca,
    y_train,
    y_test,
    train_sources,
    test_sources,
    variance
):

    header(
        "SAVING REAL 20-PC TRAIN / TEST DATASET"
    )

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if X_train_pca.shape[1] != BEST_COMPONENTS:

        raise ValueError(
            "Training PCA matrix does not contain "
            "20 components."
        )

    if X_test_pca.shape[1] != BEST_COMPONENTS:

        raise ValueError(
            "Testing PCA matrix does not contain "
            "20 components."
        )

    # --------------------------------------------------------
    # PCA column names
    # --------------------------------------------------------

    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            BEST_COMPONENTS + 1
        )
    ]

    # --------------------------------------------------------
    # Training DataFrame
    # --------------------------------------------------------

    train_df = pd.DataFrame(
        X_train_pca,
        columns=pc_columns
    )

    train_df["class"] = y_train

    train_df["source_file"] = (
        train_sources
    )

    # --------------------------------------------------------
    # Testing DataFrame
    # --------------------------------------------------------

    test_df = pd.DataFrame(
        X_test_pca,
        columns=pc_columns
    )

    test_df["class"] = y_test

    test_df["source_file"] = (
        test_sources
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_path = os.path.join(
        PC20_DIR,
        "tomato_train_20pc.csv"
    )

    test_path = os.path.join(
        PC20_DIR,
        "tomato_test_20pc.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Source-file split
    # --------------------------------------------------------

    split_rows = []

    for source, class_name in zip(
        train_sources,
        y_train
    ):

        split_rows.append({
            "source_file": source,
            "class": class_name,
            "split": "TRAIN"
        })

    for source, class_name in zip(
        test_sources,
        y_test
    ):

        split_rows.append({
            "source_file": source,
            "class": class_name,
            "split": "TEST"
        })

    split_df = (
        pd.DataFrame(
            split_rows
        )
        .drop_duplicates()
        .sort_values(
            [
                "split",
                "class",
                "source_file"
            ]
        )
    )

    split_path = os.path.join(
        PC20_DIR,
        "tomato_20pc_source_file_split.csv"
    )

    split_df.to_csv(
        split_path,
        index=False
    )

    # --------------------------------------------------------
    # PCA information
    # --------------------------------------------------------

    info_df = pd.DataFrame({

        "property": [
            "original_bands",
            "spectral_start_nm",
            "spectral_end_nm",
            "selected_components",
            "variance_retained_percent",
            "training_samples",
            "testing_samples",
            "training_source_cubes",
            "testing_source_cubes"
        ],

        "value": [
            EXPECTED_BANDS,
            WAVELENGTH_START,
            WAVELENGTH_END,
            BEST_COMPONENTS,
            variance,
            len(train_df),
            len(test_df),
            len(set(train_sources)),
            len(set(test_sources))
        ]
    })

    info_path = os.path.join(
        PC20_DIR,
        "tomato_20pc_pca_info.csv"
    )

    info_df.to_csv(
        info_path,
        index=False
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    print()
    print(
        "20-PC DATASET VERIFICATION"
    )

    print(
        f"Training shape : "
        f"{train_df.shape}"
    )

    print(
        f"Testing shape  : "
        f"{test_df.shape}"
    )

    print(
        f"PCA columns    : "
        f"{len(pc_columns)}"
    )

    print(
        f"Variance retained: "
        f"{variance:.4f}%"
    )

    print()
    print(
        "Training columns:"
    )

    print(
        list(
            train_df.columns
        )
    )

    print()
    print(
        "Training class distribution:"
    )

    print(
        train_df[
            "class"
        ].value_counts()
    )

    print()
    print(
        "Testing class distribution:"
    )

    print(
        test_df[
            "class"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Leakage check again
    # --------------------------------------------------------

    train_set = set(
        train_df[
            "source_file"
        ]
    )

    test_set = set(
        test_df[
            "source_file"
        ]
    )

    overlap = (
        train_set.intersection(
            test_set
        )
    )

    print()
    print(
        f"Source-cube overlap: "
        f"{len(overlap)}"
    )

    if overlap:

        raise ValueError(
            "Leakage detected while saving "
            "20-PC dataset."
        )

    print(
        "20-PC source-cube leakage check: PASS"
    )

    print()
    print(
        "Saved 20-PC training dataset:"
    )

    print(
        train_path
    )

    print()
    print(
        "Saved 20-PC testing dataset:"
    )

    print(
        test_path
    )

    print()
    print(
        "Saved source-file split:"
    )

    print(
        split_path
    )

    print()
    print(
        "Saved PCA information:"
    )

    print(
        info_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "TOMATO HSI REAL PCA COMPONENT SENSITIVITY"
    )

    print(
        f"Original spectral bands: "
        f"{EXPECTED_BANDS}"
    )

    print(
        f"Spectral range: "
        f"{WAVELENGTH_START}-"
        f"{WAVELENGTH_END} nm"
    )

    print(
        f"Maximum PCA components: "
        f"{MAX_PCA_COMPONENTS}"
    )

    print(
        f"Components tested: "
        f"{COMPONENTS_TO_TEST}"
    )

    print()
    print(
        "METHODOLOGY"
    )

    print(
        "PCA is fitted ONLY using training cubes."
    )

    print(
        "The test cubes remain completely unseen "
        "during PCA fitting."
    )

    print()
    print(
        "20-PC dataset will be saved separately "
        "for final Random Forest analysis."
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_files,
        test_files
    ) = create_split()

    # --------------------------------------------------------
    # Process training
    # --------------------------------------------------------

    (
        X_train_raw,
        y_train,
        train_sources
    ) = process_files(
        train_files,
        "training"
    )

    # --------------------------------------------------------
    # Process testing
    # --------------------------------------------------------

    (
        X_test_raw,
        y_test,
        test_sources
    ) = process_files(
        test_files,
        "test"
    )

    # --------------------------------------------------------
    # Dataset shapes
    # --------------------------------------------------------

    header(
        "RAW SPECTRAL DATA"
    )

    print(
        f"Training raw spectral matrix: "
        f"{X_train_raw.shape}"
    )

    print(
        f"Testing raw spectral matrix : "
        f"{X_test_raw.shape}"
    )

    print(
        f"Training labels              : "
        f"{y_train.shape}"
    )

    print(
        f"Testing labels               : "
        f"{y_test.shape}"
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    leakage_check(
        train_sources,
        test_sources
    )

    # --------------------------------------------------------
    # Mean centering
    # --------------------------------------------------------

    (
        X_train_centered,
        X_test_centered,
        training_mean
    ) = mean_center(
        X_train_raw,
        X_test_raw
    )

    # --------------------------------------------------------
    # Sensitivity experiments
    # --------------------------------------------------------

    results = []

    saved_20pc = False

    for n_components in COMPONENTS_TO_TEST:

        print()
        print(
            "-" * 75
        )

        print(
            f"EXPERIMENT: "
            f"{n_components} PCA COMPONENTS"
        )

        print(
            "-" * 75
        )

        (
            X_train_pca,
            X_test_pca,
            pca,
            variance
        ) = fit_pca(
            X_train_centered,
            X_test_centered,
            n_components
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
            f"{variance:.2f}%"
        )

        print(
            "Training Random Forest..."
        )

        model = train_random_forest(
            X_train_pca,
            y_train
        )

        predictions = model.predict(
            X_test_pca
        )

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
            *
            100
        )

        precision = (
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
            *
            100
        )

        recall = (
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
            *
            100
        )

        f1 = (
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
            *
            100
        )

        print(
            f"Accuracy        : "
            f"{accuracy:.2f}%"
        )

        print(
            f"Macro Precision : "
            f"{precision:.2f}%"
        )

        print(
            f"Macro Recall    : "
            f"{recall:.2f}%"
        )

        print(
            f"Macro F1        : "
            f"{f1:.2f}%"
        )

        results.append({

            "PCA Components":
                n_components,

            "Variance Retained (%)":
                variance,

            "Accuracy (%)":
                accuracy,

            "Macro Precision (%)":
                precision,

            "Macro Recall (%)":
                recall,

            "Macro F1 (%)":
                f1
        })

        # ----------------------------------------------------
        # Save the REAL 20-PC matrices
        # ----------------------------------------------------

        if n_components == BEST_COMPONENTS:

            save_20pc_dataset(

                X_train_pca,
                X_test_pca,

                y_train,
                y_test,

                train_sources,
                test_sources,

                variance
            )

            saved_20pc = True

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = os.path.join(
        SENSITIVITY_DIR,
        "tomato_real_pca_sensitivity_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    best_row = (
        results_df
        .loc[
            results_df[
                "Accuracy (%)"
            ].idxmax()
        ]
    )

    header(
        "PCA SENSITIVITY RESULTS"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "BEST CLASSIFICATION RESULT"
    )

    print(
        f"PCA components   : "
        f"{int(best_row['PCA Components'])}"
    )

    print(
        f"Variance retained : "
        f"{best_row['Variance Retained (%)']:.2f}%"
    )

    print(
        f"Accuracy          : "
        f"{best_row['Accuracy (%)']:.2f}%"
    )

    print(
        f"Macro Precision   : "
        f"{best_row['Macro Precision (%)']:.2f}%"
    )

    print(
        f"Macro Recall      : "
        f"{best_row['Macro Recall (%)']:.2f}%"
    )

    print(
        f"Macro F1          : "
        f"{best_row['Macro F1 (%)']:.2f}%"
    )

    # --------------------------------------------------------
    # Accuracy graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
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
        "Accuracy (%)"
    )

    plt.title(
        "Tomato HSI - PCA Component Sensitivity"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    accuracy_graph = os.path.join(
        GRAPH_DIR,
        "tomato_real_pca_component_accuracy.png"
    )

    plt.savefig(
        accuracy_graph,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Variance graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
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
        "Tomato HSI - PCA Variance Retention"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    variance_graph = os.path.join(
        GRAPH_DIR,
        "tomato_real_pca_component_variance.png"
    )

    plt.savefig(
        variance_graph,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    header(
        "FINAL 20-PC DATASET VERIFICATION"
    )

    if not saved_20pc:

        raise RuntimeError(
            "20-PC dataset was not saved."
        )

    train_20_path = os.path.join(
        PC20_DIR,
        "tomato_train_20pc.csv"
    )

    test_20_path = os.path.join(
        PC20_DIR,
        "tomato_test_20pc.csv"
    )

    train_20 = pd.read_csv(
        train_20_path
    )

    test_20 = pd.read_csv(
        test_20_path
    )

    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            21
        )
    ]

    missing_train = [
        column
        for column in pc_columns
        if column not in train_20.columns
    ]

    missing_test = [
        column
        for column in pc_columns
        if column not in test_20.columns
    ]

    if missing_train:

        raise RuntimeError(
            f"Missing training PCs: "
            f"{missing_train}"
        )

    if missing_test:

        raise RuntimeError(
            f"Missing testing PCs: "
            f"{missing_test}"
        )

    print(
        f"Training dataset: "
        f"{train_20.shape}"
    )

    print(
        f"Testing dataset : "
        f"{test_20.shape}"
    )

    print(
        "PC1-PC20 verification: PASS"
    )

    overlap = set(
        train_20[
            "source_file"
        ]
    ).intersection(
        set(
            test_20[
                "source_file"
            ]
        )
    )

    print(
        f"Final source-cube overlap: "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "Final 20-PC dataset contains "
            "source-cube leakage."
        )

    print(
        "Final leakage check: PASS"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    header(
        "REAL TOMATO PCA SENSITIVITY COMPLETED"
    )

    print(
        "Sensitivity results saved to:"
    )

    print(
        results_path
    )

    print()
    print(
        "20-PC training dataset:"
    )

    print(
        train_20_path
    )

    print()
    print(
        "20-PC testing dataset:"
    )

    print(
        test_20_path
    )

    print()
    print(
        "Graphs saved to:"
    )

    print(
        GRAPH_DIR
    )

    print()
    print(
        "The 20-PC datasets are now ready "
        "for final Random Forest analysis."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()